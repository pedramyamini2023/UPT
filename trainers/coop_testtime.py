import random

import torch
from PIL import ImageFilter
from torchvision import transforms
from tqdm import tqdm

from dassl.data.data_manager import (DataManager, DatasetWrapper,
                                     build_data_loader)
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import read_image

from .coop import CoOp

# from copy import deepcopy


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DatasetWrapper_aug(DatasetWrapper):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        super().__init__(cfg, data_source, transform, is_train)

        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.augment = augment

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                raise NotImplementedError
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    output["img"] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
                for name in ["img_aug", "img_aug2", "img_aug3"]:
                    output[name] = self._transform_image(self.augment, img0)

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output


def build_data_loader(cfg,
                      sampler_type='SequentialSampler',
                      data_source=None,
                      batch_size=64,
                      n_domain=0,
                      n_ins=2,
                      tfm=None,
                      is_train=True,
                      dataset_wrapper=None):
    # Build sampler
    if not is_train:
        random.shuffle(data_source)
    sampler = build_sampler(sampler_type,
                            cfg=cfg,
                            data_source=data_source,
                            batch_size=batch_size,
                            n_domain=n_domain,
                            n_ins=n_ins)

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager_aug(DataManager):
    def __init__(self,
                 cfg,
                 custom_tfm_train=None,
                 custom_tfm_test=None,
                 dataset_wrapper=None):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test,
                         dataset_wrapper)
        # Build test_loader
        dataset = build_dataset(cfg)

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=DatasetWrapper_aug,
        )
        self.test_loader = test_loader


def softmax_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y.softmax(1) * x.log_softmax(1)).sum(1).mean(0)


@TRAINER_REGISTRY.register()
class CoOp_testtime(CoOp):
    def build_data_loader(self):
        dm = DataManager_aug(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def test(self, split=None):
        """A generic testing pipeline."""
        # self.set_model_mode("eval")
        self.model.to(self.device)

        self._optims['prompt_learner'].param_groups[0]['lr'] = 2e-6
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        # model_state = deepcopy(self.model.state_dict())
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # self.model.load_state_dict(model_state, strict=True)
            input, input_aug, input_aug2, label = self.parse_batch_test(batch)

            for _ in range(0):
                output = self.model_inference(input)
                output_aug = self.model_inference(input_aug)
                loss = softmax_entropy(output_aug, output)
                self.model_backward_and_update(loss)

            for _ in range(0):
                output = self.model_inference(input)
                output_aug = self.model_inference(input_aug2)
                loss = softmax_entropy(output_aug, output)
                self.model_backward_and_update(loss)

            with torch.no_grad():
                output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def parse_batch_test(self, batch):
        input = batch["img"]
        input_aug = batch["img_aug"]
        input_aug2 = batch["img_aug2"]
        label = batch["label"]

        input = input.to(self.device)
        input_aug = input_aug.to(self.device)
        input_aug2 = input_aug2.to(self.device)
        label = label.to(self.device)

        return input, input_aug, input_aug2, label

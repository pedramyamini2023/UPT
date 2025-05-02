import os.path as osp
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm

from clip import clip
from dassl.data.data_manager import DataManager
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from .coop import load_clip_to_cpu
from .coop_testtime import DatasetWrapper_aug, build_data_loader
from .losses import SIMCLRLoss
from .zsclip import CUSTOM_TEMPLATES


class DataManager_aug(DataManager):
    def __init__(self,
                 cfg,
                 custom_tfm_train=None,
                 custom_tfm_test=None,
                 dataset_wrapper=None):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=DatasetWrapper_aug,
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=DatasetWrapper_aug,
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)


class CustomVPT(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)
        clip_model = clip_model.to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

        n_ctx = cfg.TRAINER.COOP.N_CTX
        dtype = clip_model.dtype
        ctx_dim = self.clip_model.visual.positional_embedding.shape[-1]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.visual_ctx = nn.Parameter(ctx_vectors)
        # self.mlp = self._build_mlp(768, 4096, 256, dtype=dtype)
        self.mlp = self._build_mlp(768, 512, 256, dtype=dtype)
        self.mlp.to(device)

    def _build_mlp(self, in_dim, mlp_dim, out_dim, dtype):
        return nn.Sequential(
            OrderedDict([
                ("layer1", nn.Linear(in_dim, mlp_dim).to(dtype)),
                # ("bn1", nn.SyncBatchNorm(mlp_dim)),
                ("relu1", nn.ReLU(inplace=True)),
                # ("layer2", nn.Linear(mlp_dim, mlp_dim).to(dtype)),
                # ("bn2", nn.SyncBatchNorm(mlp_dim)),
                # ("relu2", nn.ReLU(inplace=True)),
                ("layer3", nn.Linear(mlp_dim, out_dim).to(dtype)),
            ]))

    def forward(self, image, aug1, aug2, training=True):
        if training:
            image_features, _ = self.clip_model.encode_image_prompt(
                image, self.visual_ctx)
            image_features = image_features / image_features.norm(dim=-1,
                                                                  keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.t()
        else:
            logits = None

        _, h2 = self.clip_model.encode_image_prompt(aug1, self.visual_ctx)
        _, h3 = self.clip_model.encode_image_prompt(aug2, self.visual_ctx)
        aug1_embed = self.mlp(h2)
        aug2_embed = self.mlp(h3)

        return logits, aug1_embed, aug2_embed


@TRAINER_REGISTRY.register()
class VPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom VPT")
        self.model = CustomVPT(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # if "ctx" not in name:
            if "clip" in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            # load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = 1  # torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
            )
            self.model = nn.DataParallel(self.model)

        self.ssl_loss = SIMCLRLoss()

    def forward_backward(self, batch):
        image, image_aug, image_aug2, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, aug1_embed, aug2_embed = self.model(
                    image, image_aug, image_aug2)
                loss_ce = F.cross_entropy(output, label)
                loss_ssl, acc_ssl = self.ssl_loss(aug1_embed, aug2_embed)
                loss = loss_ce + loss_ssl * 0.0
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, aug1_embed, aug2_embed = self.model(
                image, image_aug, image_aug2)
            loss_ce = F.cross_entropy(output, label)
            loss_ssl, acc_ssl = self.ssl_loss(aug1_embed, aug2_embed)
            loss = loss_ce + loss_ssl * 0.0
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_ssl": loss_ssl.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "acc_ssl": acc_ssl.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def parse_batch_train(self, batch):
        input = batch["img"]
        input_aug = batch["img_aug"]
        input_aug2 = batch["img_aug2"]
        label = batch["label"]
        input = input.to(self.device)
        input_aug = input_aug.to(self.device)
        input_aug2 = input_aug2.to(self.device)
        label = label.to(self.device)
        return input, input_aug, input_aug2, label

    # test-time training

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

        self._optims['model'].param_groups[0]['lr'] = 2e-4
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        model_state = deepcopy(self.model.state_dict())
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            self.model.load_state_dict(model_state, strict=True)
            input, input_aug, input_aug2, input_aug3, label = self.parse_batch_test(
                batch)

            for _ in range(0):
                logits, aug1_embed, aug2_embed = self.model_inference(
                    input, input_aug, input_aug2, training=False)
                loss_ssl, _ = self.ssl_loss(aug1_embed, aug2_embed)
                self.model_backward_and_update(loss_ssl)

            for _ in range(0):
                logits, aug1_embed, aug2_embed = self.model_inference(
                    input, input_aug2, input_aug3, training=False)
                loss_ssl, _ = self.ssl_loss(aug1_embed, aug2_embed)
                self.model_backward_and_update(loss_ssl)

            with torch.no_grad():
                output, _, _ = self.model_inference(input,
                                                    input_aug,
                                                    input_aug2,
                                                    training=True)
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input, input_aug, input_aug2, training=True):
        return self.model(input, input_aug, input_aug2, training=training)

    def parse_batch_test(self, batch):
        input = batch["img"]
        input_aug = batch["img_aug"]
        input_aug2 = batch["img_aug2"]
        input_aug3 = batch["img_aug3"]
        label = batch["label"]
        input = input.to(self.device)
        input_aug = input_aug.to(self.device)
        input_aug2 = input_aug2.to(self.device)
        input_aug3 = input_aug3.to(self.device)
        label = label.to(self.device)
        return input, input_aug, input_aug2, input_aug3, label

rm -rf `find -type d -name .ipynb_checkpoints`
yapf -r -i ./*.py
isort -rc ./*.py
flake8 ./*.py

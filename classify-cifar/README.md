<h1 align="center">
Fromage 🧀 optimiser
</h1>

## CIFAR-10 classification experiments

This codebase is based on one by [kuangliu](https://github.com/kuangliu/pytorch-cifar). To run the training script, use a command like:
```
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 2048
```
The `resnet-18` model architecture is given in the `models/` directory. We provide the shell script `batch.sh` to run multiple experiments.

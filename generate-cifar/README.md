<h1 align="center">
Fromage 🧀 optimiser
</h1>

## CIFAR-10 class conditional GAN

The following Python packages are required: numpy, torch, torchvision, tqdm.

An example job is
```
python main.py --seed 0 --optim fromage --initial_lr 0.01
```
See inside `sh batch.sh` for the settings used in the paper.

## Acknowledgements
- The self attention block implementation is originally by https://github.com/zhaoyuzhi.
- The FID score implementation is by https://github.com/mseitzer/pytorch-fid.
- Thanks also go to [Jiahui Yu](https://jiahuiyu.com/).

## License
This repository (exluding the `fid/` subdirectory) is made available under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

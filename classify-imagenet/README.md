 ## Requirements
 - [PyTorch](http://pytorch.org)
 - [NVIDIA APEX](https://github.com/NVIDIA/apex#quick-start)
 
## Data Preparation
Download the ImageNet 2012 dataset and structure the dataset under 
train and val subfloders. You can follow [this page](https://github.com/pytorch/examples/tree/master/imagenet#requirements) 
to structure the dataset. The data directory should be in the form:

    data/
        ├── train/
            ├── n01440764/
            ├── n01443537/
            ├── ...
        ├── val/
            ├── n01440764/
            ├── n01443537/
            ├── ...        
 
 ## COMMANDS
```
cd classify-imagenet
python -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py --data $DATA_DIR --results_dir $RESULTS_DIR \
--save $EXPR_NAME --optimizer fromage --learning_rate 1e-2 --seed 0
```
Above `$DATA_DIR` refers to the dataset directory path, `$RESULTS_DIR` is the results directory with `$EXPR_NAME` giving
a name for the experiment.
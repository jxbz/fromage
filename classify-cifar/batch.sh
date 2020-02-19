#!/bin/bash

python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 2048
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 1024
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 512
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 256
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 128

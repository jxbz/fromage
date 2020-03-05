#!/bin/sh

python main.py --optim fromage --lrG 0.01 --lrD 0.01 --epochs 121 --seed 0
python main.py --optim fromage --lrG 0.01 --lrD 0.01 --epochs 121 --seed 1
python main.py --optim fromage --lrG 0.01 --lrD 0.01 --epochs 121 --seed 2

python main.py --optim sgd --lrG 0.01 --lrD 0.01 --epochs 121 --seed 0
python main.py --optim sgd --lrG 0.01 --lrD 0.01 --epochs 121 --seed 1
python main.py --optim sgd --lrG 0.01 --lrD 0.01 --epochs 121 --seed 2

python main.py --optim adam --lrG 0.0001 --lrD 0.0004 --epochs 121 --seed 0
python main.py --optim adam --lrG 0.0001 --lrD 0.0004 --epochs 121 --seed 1
python main.py --optim adam --lrG 0.0001 --lrD 0.0004 --epochs 121 --seed 2
# Tune learning rates in each optimiser
python main.py --seed 0 --optim adam --initial_lr 1.0
python main.py --seed 0 --optim adam --initial_lr 0.1
python main.py --seed 0 --optim adam --initial_lr 0.01
python main.py --seed 0 --optim adam --initial_lr 0.001
python main.py --seed 0 --optim adam --initial_lr 0.0001
python main.py --seed 0 --optim adam --initial_lr 0.00001

python main.py --seed 0 --optim sgd --initial_lr 1.0
python main.py --seed 0 --optim sgd --initial_lr 0.1
python main.py --seed 0 --optim sgd --initial_lr 0.01
python main.py --seed 0 --optim sgd --initial_lr 0.001
python main.py --seed 0 --optim sgd --initial_lr 0.0001

python main.py --seed 0 --optim fromage --initial_lr 1.0
python main.py --seed 0 --optim fromage --initial_lr 0.1
python main.py --seed 0 --optim fromage --initial_lr 0.01
python main.py --seed 0 --optim fromage --initial_lr 0.001
python main.py --seed 0 --optim fromage --initial_lr 0.0001

# Run different random seeds for best setting
python main.py --seed 1 --optim fromage --initial_lr 0.01
python main.py --seed 1 --optim adam --initial_lr 0.0001
python main.py --seed 1 --optim sgd --initial_lr 0.01

python main.py --seed 2 --optim fromage --initial_lr 0.01
python main.py --seed 2 --optim adam --initial_lr 0.0001
python main.py --seed 2 --optim sgd --initial_lr 0.01

# LARS experiment
python main.py --seed 0 --optim lars --initial_lr 0.01
python main.py --seed 1 --optim lars --initial_lr 0.01
python main.py --seed 2 --optim lars --initial_lr 0.01
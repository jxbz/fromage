# Tune p_bound (constrains parameters to a bounded set)
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 128
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 128 --p_bound 1.0
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 128 --p_bound 2.0
python main.py --optim fromage --lr 0.01 --seed 1337 --bsz 128 --p_bound 3.0

# Tune learning rates in each algorithm
python main.py --optim sgd --lr 1.0 --seed 1337 --bsz 128
python main.py --optim sgd --lr 0.1 --seed 1337 --bsz 128
python main.py --optim sgd --lr 0.01 --seed 1337 --bsz 128
python main.py --optim sgd --lr 0.001 --seed 1337 --bsz 128
python main.py --optim sgd --lr 0.0001 --seed 1337 --bsz 128

python main.py --optim adam --lr 1.0 --seed 1337 --bsz 128
python main.py --optim adam --lr 0.1 --seed 1337 --bsz 128
python main.py --optim adam --lr 0.01 --seed 1337 --bsz 128
python main.py --optim adam --lr 0.001 --seed 1337 --bsz 128
python main.py --optim adam --lr 0.0001 --seed 1337 --bsz 128
python main.py --optim adam --lr 0.00001 --seed 1337 --bsz 128

python main.py --optim fromage --lr 1.0 --seed 1337 --bsz 128 --p_bound 1.0
python main.py --optim fromage --lr 0.1 --seed 1337 --bsz 128 --p_bound 1.0
python main.py --optim fromage --lr 0.001 --seed 1337 --bsz 128 --p_bound 1.0
python main.py --optim fromage --lr 0.0001 --seed 1337 --bsz 128 --p_bound 1.0

# Run different random seeds for best runs
python main.py --optim fromage --lr 0.01 --seed 0 --bsz 128 --p_bound 1.0
python main.py --optim fromage --lr 0.01 --seed 1 --bsz 128 --p_bound 1.0

python main.py --optim sgd --lr 0.1 --seed 0 --bsz 128
python main.py --optim sgd --lr 0.1 --seed 1 --bsz 128

python main.py --optim adam --lr 0.001 --seed 0 --bsz 128
python main.py --optim adam --lr 0.001 --seed 1 --bsz 128
# Tune p_bound in Fromage (this prevents overfitting)
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --p_bound 1.0
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --p_bound 3.0

# Tune lr in SGD
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.0001
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.001
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 0.1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 10

# Tune lr in Adam
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 1.0
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.01
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.00001

# Run different random seeds for best runs
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --p_bound 1.0 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --p_bound 1.0 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim sgd --lr 1 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.0001 --seed 2
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001 --seed 1
python main.py --cuda --epochs 20 --model Transformer --optim adam --lr 0.001 --seed 2

# Fromage tune LR
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 1.0 --p_bound 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.1 --p_bound 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.001 --p_bound 1.0
python main.py --cuda --epochs 10 --model Transformer --optim fromage --lr 0.0001 --p_bound 1.0
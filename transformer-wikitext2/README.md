<h1 align="center">
Fromage 🧀 optimiser
</h1>

## Transformer training on Wikitext-2

This codebase is from the [Pytorch example](https://github.com/pytorch/examples/tree/master/word_language_model). To run the training script, use a command like:
```
python main.py --cuda --epochs 20 --model Transformer --optim fromage --lr 0.01 --p_bound 1.0
```
We provide the shell script `batch.sh` to run multiple experiments.

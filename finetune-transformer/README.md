<h1 align="center">
Fromage 🧀 optimiser
</h1>

## Fine-tune a transformer

In the directory `transformers` you will find our modification of the [🤗 Transformers](https://github.com/huggingface/transformers) repository.

Our modifications are limited to:
1. the modification of `transformers/examples/run_squad.py` to include Fromage.
2. putting the SQuAD data into `transformers/data/squad1`.
3. the addition of a launcher script `transformers/examples/batch.sh` to launch our training runs.

## Quick start

To get up and running, you will probably want to create a new virtual Python environment using `conda` as follows:
```
conda create -n hugs python=3.6
conda activate hugs
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
Next, change directory to `transformers` and then run:
```
pip install .
pip install -r ./examples/requirements.txt
```
You should now be able to switch directory to `transformers/examples` and to reproduce our experiments run:
```
sh batch.sh
```


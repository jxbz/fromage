<h1 align="center">
Fromage 🧀 optimiser
</h1>

## Generate all plots

This subdirectory contains a Jupyter notebook `generate_plots.ipynb` that will generate all figures in the paper.
It looks in the directory `/make-plots/logs/` for the training log files, and it spits out plots into the directory `/make-plots/figures/`.

If you want to see how the `logs` folder got populated, you will need to check out our training scripts. Go [a level up](../../../tree/master) in the repository.

To run the notebook you will need to install Python and Jupyter Notebook. The notebook relies on the following Python modules:

```
import matplotlib
import numpy
import tensorboard
import pickle
```

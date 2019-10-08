# Bayesian Sparse Deep Learning
Experiment code for "An Adaptive Empirical Bayesian Method for Sparse Deep Learning"


## Regression: UCI dataset

### Requirement
* Python 2.7
* [PyTorch > 1.1](https://pytorch.org/)
* numpy



## Classification: Sparse Residual Network
### Requirement
* Python 2.7
* [PyTorch > 1.1](https://pytorch.org/)
* numpy
* CUDA

### Pretrain a dense model
```{python}
python bayes_cnn.py -lr 0.1 -invT 1e9 -save 1 -finetune -1  
```

### Finetune a sparse model with 90% sparsity through stochastic approximation
```{python}
python bayes_cnn.py -lr 1e-4 -invT 1e9 -v0 0.005 -v1 1e-5 -sparse 0.9 -method sa
```

For other sparse rates, one needs to tune the best v0 and v1 parameters, e.g. 30%: (0.5, 1e-3), 50%: (0.1, 5e-4), 70%: (0.1, 5e-5).

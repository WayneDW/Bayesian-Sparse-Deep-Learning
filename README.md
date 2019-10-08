# Bayesian Sparse Deep Learning
Experiment code for "An Adaptive Empirical Bayesian Method for Sparse Deep Learning". The key idea is to show that stochastic approximation is more robust than expectation-maximization (EM) to estimate latent variables.

## Large-p-small-n Linear Regression

![GitHub Logo](/figures/lr_simulation.png)


## Regression: UCI dataset

### Requirement
* Python 2.7
* [PyTorch > 1.1](https://pytorch.org/)
* numpy

Since the model is simple, GPU environment doesn't give you significant computational accelerations. Therefore, we use CPU instead. 

The followings are the commands to run the different methods (stochastic approximation SGHMC/EM-SGHMC/vanilla SGHMC) on the [Boston housing price](https://www.kaggle.com/vikrishnan/boston-house-prices) dataset.
```{python}
python uci_run.py -data boston -c sa -invT 1 -v0 0.1 -anneal 1.003 -seed 5
python uci_run.py -data boston -c em -invT 1 -v0 0.1 -anneal 1.003 -seed 5
python uci_run.py -data boston -c sghmc -invT 1 -v0 0.1 -anneal 1.003 -seed 5
```

You can also use the other datasets to test the performance, e.g. yacht, energy-efficiency, wine and concrete.


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

### Finetune a sparse model through stochastic approximation
```{python}
python bayes_cnn.py -lr 1e-4 -invT 1e9 -v0 0.005 -v1 1e-5 -sparse 0.9 -method sa
```
The default code can produce a 90%-sparsity Resnet20 model with 91.4% accuracy


For other sparse rates, one needs to tune the best v0 and v1 parameters, e.g. 30%: (0.5, 1e-3), 50%: (0.1, 5e-4), 70%: (0.1, 5e-5).



## References:

Wei Deng, Xiao Zhang, Faming Liang, Guang Lin, An Adaptive Empirical Bayesian Method for Sparse Deep Learning, NIPS, 2019

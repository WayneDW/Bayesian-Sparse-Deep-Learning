# Bayesian Sparse Deep Learning
Experiment code for "[An Adaptive Empirical Bayesian Method for Sparse Deep Learning](https://arxiv.org/pdf/1910.10791.pdf)". We propose a novel adaptive empirical Bayesian method to efficiently train hierarchical Bayesian mixture DNN models, where the parameters are learned through sampling while the priors are learned through optimization. In addition, this model can be further generalized to a class of adaptive sampling algorithms for estimating various state-space models in deep learning.

```
@inproceedings{deng2019,
  title={An Adaptive Empirical Bayesian Method for Sparse Deep Learning},
  author={Wei Deng and Xiao Zhang and Faming Liang and Guang Lin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Large-p-small-n Linear Regression

![GitHub Logo](/figures/lr_simulation.png)


## Regression: UCI dataset

### Requirement
* [PyTorch > 1.01](https://pytorch.org/)
* Python 2.7
* numpy
* sklearn

Since the model is simple, GPU environment doesn't give you significant computational accelerations. Therefore, we use CPU instead. 

The followings are the commands to run the different methods (stochastic approximation SGHMC/EM-SGHMC/vanilla SGHMC) on the [Boston housing price](https://www.kaggle.com/vikrishnan/boston-house-prices) dataset.
```{python}
python uci_run.py -data boston -c sa -invT 1 -v0 0.1 -anneal 1.003 -seed 5
python uci_run.py -data boston -c em -invT 1 -v0 0.1 -anneal 1.003 -seed 5
python uci_run.py -data boston -c sghmc -invT 1 -v0 0.1 -anneal 1.003 -seed 5
```

You can also use the other datasets to test the performance, e.g. yacht, energy-efficiency, wine and concrete. To obtain a comprehensive evaluation, you may need to try many different seeds.


## Classification: MNIST/Fashion MNIST

You can adjust the posterior_cnn.py and use the model in ./model/model_zoo_mnist.py. **99.7x%** results on MNIST dataset can be easily obtained with the hyperparameters (most importantly: **temperature**) in the paper. To run the Adversarial examples, you can include the file in tools/attacker.py and make the corresponding changes.

## Classification: Sparse Residual Network on CIFAR10
### Requirement
* Python 2.7
* [PyTorch > 1.01](https://pytorch.org/)
* numpy
* CUDA

### Pretrain a dense model
```{python}
python bayes_cnn.py -lr 2e-6 -invT 20000 -save 1 -prune 0  
```

### Finetune a sparse model through stochastic approximation
```{python}
python bayes_cnn.py -lr 2e-9 -invT 1000 -anneal 1.005 -v0 0.005 -v1 1e-5 -sparse 0.9 -c sa -prune 1
```
The default code can produce a **90%-sparsity Resnet20 model with the state-of-the-art 91.56% accuracy (PyTorch version 1.01) based on 27K parameters**, by contrast, EM-based SGHMC (with step size 1) and vanilla SGHMC algorithm (with step size 0) obtain much worse results. The running log based on the default seed is saved in the output folder, you can try other seeds to obtain higher results.


For other sparse rates, one needs to tune the best v0 and v1 parameters, e.g. 30%: (0.5, 1e-3), 50%: (0.1, 5e-4), 70%: (0.1, 5e-5), to achive the state-of-the-art.



## References:

Wei Deng, Xiao Zhang, Faming Liang, Guang Lin, [An Adaptive Empirical Bayesian Method for Sparse Deep Learning](https://arxiv.org/pdf/1910.10791.pdf), NeurIPS, 2019

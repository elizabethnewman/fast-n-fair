# fast-n-fair
Fast adversarial training for fair machine learning

# Installation

We can install using ```pip```
```console
!python -m pip install git+https://github.com/elizabethnewman/fast-n-fair.git
```
or clone from Github directly
```console
git clone https://github.com/elizabethnewman/fast-n-fair.git
```
# Getting Started

First, let's install the necessary packages:
```python 
import matplotlib.pyplot as plt
import torch.optim

from fastNfair.data import generate_unfair_data, visualize_unfair_data
from fastNfair.objective_functions import ObjectiveFunctionLogisticRegression
from fastNfair.training import TrainerSGD, Evaluator
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
```

Next, let's create the data:
```python
# for reproducibility
torch.manual_seed(42)

# generate data
x_train, y_train, s_train = generate_unfair_data(250)

n_train = 200
x_val, y_val, s_val = x_train[n_train:], y_train[n_train:], s_train[n_train:]
x_train, y_train, s_train = x_train[:n_train], y_train[:n_train], s_train[:n_train]

# test data
x_test, y_test, s_test = generate_unfair_data(50, p1=p1, p2=p2, alpha=alpha)


visualize_unfair_data((x_train, y_train, s_train), domain=(-0.1, 1.1, -0.1, 1.1))
plt.show()
```

Now, let's train!
```
# for reproducibility
torch.manual_seed(42)

# create linear network
my_net = net.NN(lay.singleLayer(2, 1, act=act.identityActivation(), bias=True))

# create objective function
fctn = ObjectiveFunctionLogisticRegression(my_net)

# choose optimizer
opt = torch.optim.Adam(fctn.parameters(), lr=1e-2)

# construct trainer
trainer = TrainerSGD(opt, max_epochs=20)

# train!
results_train = trainer.train(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test), 
                              verbose=True, robust=False)
```
Finally, we can compute the numerical results as folows:
```python 
evaluator = Evaluator()
results_eval = evaluator.evaluate(fctn, (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test))
```
That's it!  To do robust training, just change the train flag to ```robust=True```.

# Example Notebooks

TODO: add more description and remove personal access token when repository is published

[Toy Example](https://github.com/elizabethnewman/fast-n-fair/blob/main/fastNfair/examples/notebooks/FastNFair_ToyExample.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/fast-n-fair/blob/main/fastNfair/examples/notebooks/FastNFair_ToyExample.ipynb)

[Toy Example](https://github.com/elizabethnewman/fast-n-fair/blob/main/fastNfair/examples/notebooks/FastNFair_MNISTBinary.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elizabethnewman/fast-n-fair/blob/main/fastNfair/examples/notebooks/FastNFair_MNISTBinary.ipynb)

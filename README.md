# levmarq-torch

A basic PyTorch implementation of the Levenberg-Marquardt algorithm. This solves minimization problems of the form

$$\mathbf{x}^* = \mathrm{argmin}_{\mathbf{x}} |\mathbf{y} - \mathbf{\hat{y}}(\mathbf{x})|^2 \, .$$

The implementation is batched over the parameters $\mathbf{x}$ and datapoints $\mathbf{y}$.

Based on implementation 1 from [Gavin 2022](https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf)
and some help from [Connor Stone](https://github.com/ConnorStoneAstro/).

## Installation

Running
```
git clone git@github.com:adam-coogan/levmarq-torch.git
cd levmarq-torch
pip install .
```
will install the `levmarq_torch` package.

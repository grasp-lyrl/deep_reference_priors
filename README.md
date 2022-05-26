# Deep Reference priors

[Deep Reference priors](https://arxiv.org/abs/2202.00187) pre-train the weights
of a neural network with unlabeled data, using the theory of reference priors.
It allows the data to dominate the posterior rather than the choice of the
prior. The prior is represented by a finite number of particles (or neural
networks) which are trained to span the prediction space.

This method is competitive with other SoTA semi-supervised learning methods.

![](./assets/ref_prior_table.png =x300)

## Setup and Usage

```
conda create -n ref_prior --file setup.yml
conda activate ref_prior
python train.py
```

## Acknowledgements 
Parts of this code are adapted from https://github.com/kekmodel/FixMatch-pytorch

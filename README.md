# Deep Reference priors

Deep reference priors (https://arxiv.org/abs/2202.00187) are used to pre-train
the weights of a neural network using unlabeled data. The theory of reference priors 
dictates that the prior distribution is supported on a finite number of
weights. Hence, we represent the prior as an ensemble of $K$ neural networks.

Deep reference priors are trained based on the theory of reference priors which
is an uninformative prior. It allows the data to dominate the posterior rather
than the choice of the prior. Intuitively, Deep reference priors encourage a
finite number of particles to span the prediction space.


Parts of this code are adapted from https://github.com/kekmodel/FixMatch-pytorch

# Deep Reference priors

Deep reference priors are used to pre-train the weights of a neural network
using unlabeled data. The prior distribution is discrete and is supported on a
finite number of weights. Hence, we represent the prior as a set (or ensemble)
of $K$ neural network.

Deep reference priors are trained based on the theory of reference priors which
is an uninformative prior. It allows the data to dominate the determine the
posterior rather than the choice of the prior. Intuitively, Deep reference
priors encourages a finite number of particles to span the prediction space.

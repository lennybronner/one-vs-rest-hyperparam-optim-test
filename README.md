# one-vs-rest-hyperparam-optim-test
Here I tested a different way of doing hyperparameter optimization in one-vs-rest classifiers. 

The idea is that `sklearn.multiclass.OneVsRestClassifier` uses the same hyperparameter value for all base estimators. I was curious what happens if you allowed every estimator to have it's own hyperparameter value.

It's quite likely that this is theoretically unsound, but I was curious. Empirically, it doesn't seem to help, though I only tried it on an example that isn't great.

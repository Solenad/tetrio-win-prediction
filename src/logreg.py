import numpy as np
import comet_ml
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss


def fit_minibatch_sgd(model, X_train, y_train, epochs, batch_size, random_state):
    n_samples = X_train.shape[0]
    classes = np.unique(y_train)
    rng = np.random.default_rng(random_state)
    exp = comet_ml.get_global_experiment()

    loss_history = []
    global_step = 0

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            if epoch == 0 and start == 0:
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)

            global_step += 1

        current_proba = model.predict_proba(X_train)
        epoch_loss = log_loss(y_train, current_proba, labels=classes)
        loss_history.append(epoch_loss)

        if exp:
            exp.log_metric("epoch_loss", epoch_loss, step=epoch)

    return model, loss_history  # <--- NEW: Return the history


class MinibatchSGDWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, eta0=0.001, epochs=200, batch_size=32, random_state=1):
        self.eta0 = eta0
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.model = SGDClassifier(
            loss="log_loss",
            eta0=self.eta0,
            learning_rate="constant",
            random_state=self.random_state,
        )

        self.model, self.loss_history_ = fit_minibatch_sgd(  # <--- NEW
            self.model,
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self.classes_ = np.unique(y)
        self.coef_ = self.model.coef_
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X))

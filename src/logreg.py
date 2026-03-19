import numpy as np
import comet_ml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score


def fit_minibatch_sgd(model, X_train, y_train, epochs, batch_size, random_state):
    n_samples = X_train.shape[0]
    classes = np.unique(y_train)
    rng = np.random.default_rng(random_state)

    exp = comet_ml.get_global_experiment()

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            X_batch = X_train.iloc[batch_idx]
            y_batch = y_train.iloc[batch_idx]

            if epoch == 0 and start == 0:
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)

        if exp:
            y_proba = model.predict_proba(X_train)
            y_pred = model.predict(X_train)

            epoch_loss = log_loss(y_train, y_proba, labels=classes)
            epoch_acc = accuracy_score(y_train, y_pred)

            exp.log_metric("epoch_loss", epoch_loss, step=epoch)
            exp.log_metric("epoch_accuracy", epoch_acc, step=epoch)

    return model


class MinibatchSGDWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, eta0=0.001, epochs=200, batch_size=32, random_state=1):
        self.eta0 = eta0
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):

        self.model = SGDClassifier(
            loss="log_loss",
            eta0=self.eta0,
            learning_rate="constant",
            random_state=self.random_state,
        )

        self.model = fit_minibatch_sgd(
            self.model,
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self.classes_ = np.unique(y)
        self.coef_ = self.model.coef_  # Expose coef_ for feature importance
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

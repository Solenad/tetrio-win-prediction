import numpy as np


def fit_minibatch_sgd(model, X_train, y_train, epochs, batch_size, random_state):

    n_samples = X_train.shape[0]
    classes = np.unique(y_train)
    rng = np.random.default_rng(random_state)

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
    return model

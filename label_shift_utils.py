import numpy as np


def get_dirichlet_marginal(alpha, seed):
    np.random.seed(seed)

    return np.random.dirichlet(alpha)


def get_resampled_indices(y, num_labels, Py, seed):
    np.random.seed(seed)
    # get indices for each label
    indices_by_label = [(y == k).nonzero()[0] for k in range(num_labels)]
    num_samples = int(
        min([len(indices_by_label[i]) / Py[i] for i in range(num_labels)])
    )

    agg_idx = []
    for i in range(num_labels):
        # sample an example from X with replacement
        idx = np.random.choice(
            indices_by_label[i], size=int(num_samples * Py[i]), replace=False
        )
        agg_idx.append(idx)

    return np.concatenate(agg_idx)
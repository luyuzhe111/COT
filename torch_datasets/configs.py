

def get_train_val_size(dataset):
    config = {
        'cifar-10': [50000, 10000],
        'cifar-100': [50000, 10000],
        'tiny-imagenet': [90000, 10000]
    }

    return config[dataset]

def get_expected_label_distribution(dataset):
    config = {
        'cifar-10': [1 / 10] * 10,
        'cifar-100': [1 / 100] * 100,
        'tiny-imagenet': [1 / 200] * 200
    }

    return config[dataset]
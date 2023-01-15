

def get_train_val_size(dataset):
    config = {
        'cifar-10': [50000, 10000],
        'cifar-100': [50000, 10000],
        'tiny-imagenet': [90000, 10000]
    }

    return config[dataset]
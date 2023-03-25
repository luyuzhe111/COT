from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
import torchvision.transforms as transforms


def get_breeds_dataset(data_dir, split, transform):
    ret = make_living17(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0]) 
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    if split == 'train':
        dataset = folder.ImageFolder(
            root=f"{data_dir}/train/", 
            transform = transform, 
            label_mapping = source_label_mapping
        )
    elif split == 'test':
        dataset =  folder.ImageFolder(
            root=f"{data_dir}/train/", 
            transform = transform, 
            label_mapping = target_label_mapping
        )
    else:
        raise ValueError('unknown split')
    
    dataset.data = dataset.imgs # to have the same attributes as torch.Dataset

    return dataset
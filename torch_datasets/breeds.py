from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
import os


def get_breeds_dataset(data_dir, dsname, subpopulation, split, transform, corr='clean', corr_sev=0):
    if dsname == 'Living-17':
        ret = make_living17(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif dsname == 'Nonliving-26':
        ret = make_nonliving26(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif dsname == 'Entity-13':
        ret = make_entity13(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif dsname == 'Entity-30':
        ret = make_entity30(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    else:
        raise ValueError(f'unknown dataset: {dsname}')
    
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0]) 
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])
    assert subpopulation in ['same', 'novel'], 'unknown subpopulation'

    if split == 'train':
        dataset = folder.ImageFolder(
            root=f"{data_dir}/imagenetv1/train/", 
            transform = transform, 
            label_mapping = source_label_mapping
        )
    elif split == 'test':
        dataset =  folder.ImageFolder(
            root=f"{data_dir}/imagenetv1/val/" if corr == 'clean' else f"{data_dir}/imagenet-c/{corr}/{corr_sev}", 
            transform = transform, 
            label_mapping = ( source_label_mapping if subpopulation == 'same' else target_label_mapping)
        )
    else:
        raise ValueError('unknown split')
    
    dataset.data = dataset.imgs # to have the same attributes as torch.Dataset

    return dataset
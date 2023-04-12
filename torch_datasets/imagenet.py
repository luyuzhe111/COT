from torchvision.datasets import ImageFolder


def get_imagenet_dataset(data_dir, subpopulation, transform, corr='clean', corr_sev=0):
    if subpopulation == 'natural':
        if corr_sev == 0:
            return ImageFolder(root=f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/", transform=transform)
        elif corr_sev == 1:
            return ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform=transform)
        elif corr_sev == 2:
            return ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform=transform)
        elif corr_sev == 3:
            return ImageFolder(f"{data_dir}/imagenet-sketch/", transform=transform)
        
    elif subpopulation == 'same':
        assert corr_sev > 0, 'corr sev should > 0 for synthetic shifts'
        return ImageFolder(root=f"{data_dir}/imagenet-c/{corr}/{corr_sev}", transform=transform)
        
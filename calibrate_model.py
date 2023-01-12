import argparse
import torchvision.models as models
import torch.nn as nn
import json
from misc.temperature_scaling import ModelWithTemperature
from load_data import *
from model import ResNet18, ResNet50, VGG11
from utils import gather_outputs

"""# Configuration"""
parser = argparse.ArgumentParser(description='Calibrate Model')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--data_path', default='./data/CIFAR-100/', type=str)
parser.add_argument('--corruption_path', default='./data/CIFAR-100-C/', type=str)
parser.add_argument('--data_type', default='cifar-100', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--model_seed', default="1", type=str)
parser.add_argument('--seed', default=1, type=int)
args = vars(parser.parse_args())


def calibrate(model, valloader):
    model.eval()
    scaled_model = ModelWithTemperature(model)
    scaled_model.find_temperature(valloader)
    return scaled_model, scaled_model.temperature


def main():
    data_type = args['data_type']
    
    save_dir_path = f"./checkpoints/{data_type}/{args['arch']}"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup train/val_iid loaders
    trainset, valset = load_image_dataset(corruption_type='clean',
                                          clean_path=args['data_path'],
                                          corruption_path=args['corruption_path'],
                                          corruption_severity=0,
                                          type=data_type,
                                          datatype='train')
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=args['batch_size'], shuffle=True)

    main_model_ckpt = f"{save_dir_path}/base_model_{args['model_seed']}.pt"
    alt_model_ckpt = f"{save_dir_path}/base_model_alt.pt"

    # init and train base model
    if args['arch'] == 'resnet18':
        main_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
        alt_model = ResNet18(num_classes=args['num_classes'], seed=114514).cuda()
    elif args['arch'] == 'resnet50':
        main_model = ResNet50(num_classes=args['num_classes'], seed=args['seed']).cuda()
        alt_model = ResNet50(num_classes=args['num_classes'], seed=114514).cuda()
    elif args['arch'] == 'vgg11':
        main_model = VGG11(num_classes=args['num_classes'], seed=args['seed']).cuda()
        alt_model = VGG11(num_classes=args['num_classes'], seed=114514).cuda()
    else:
        raise ValueError('incorrect model name')
    
    main_model = torch.load(main_model_ckpt, map_location=device)
    alt_model = torch.load(alt_model_ckpt, map_location=device)

    main_model, main_t = calibrate(main_model, valloader)

    iid_acts, iid_preds, iid_tars = gather_outputs(main_model, valloader, device, './misc/test_100.pkl')
    act = nn.Softmax(dim=1)

    # acc & average confidence should be similar after calibration
    # if acc >> conf, then the model is still overconfident, try increasing the num of optimization steps in the calibator
    # if acc << conf, the the model is underconfident / misspecified, this means the model is under trained. try training 
    # the model more
    print('acc:', (iid_preds == iid_tars).float().mean())
    print('average confidence:', act(iid_acts).amax(1).mean().item())



if __name__ == "__main__":
    main()
import argparse

from projnorm import *
from load_data import *
from model import ResNet18, ResNet50, VGG11
from utils import evaluation
import json

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
parser.add_argument('--corruption_path', default='./data/CIFAR-10-C', type=str)
parser.add_argument('--data_type', default='cifar-10', type=str)
parser.add_argument('--corruption', default='snow', type=str)
parser.add_argument('--severity', default=5, type=int)
parser.add_argument('--pseudo_iters', default=1, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_ood_samples', default=10000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--model_seed', default="1", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_base_model', action='store_true', default=False, help='use base or ref model')
args = vars(parser.parse_args())

print(args)

if __name__ == "__main__":
    # setup valset_iid/val_ood loaders
    torch.manual_seed(args['seed'])
    random_seeds = torch.randint(0, 10000, (2,))

    print('random seeds:', random_seeds)
    
    data_type = args['data_type']

    n_ood_sample = args['num_ood_samples']

    print('num of ood samples:', n_ood_sample)
    print('pseudo iters:', args['pseudo_iters'])

    valset_ood = load_image_dataset(corruption_type=args['corruption'],
                                      clean_path=args['data_path'],
                                      corruption_path=args['corruption_path'],
                                      corruption_severity=args['severity'],
                                      datatype='test',
                                      num_samples=n_ood_sample,
                                      type=data_type,
                                      seed=random_seeds[1])

    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=args['batch_size'], shuffle=True)

    # init ProjNorm
    save_dir_path = f"./checkpoints/{data_type}/{args['arch']}"

    base_model = torch.load(f"{save_dir_path}/base_model_{args['model_seed']}.pt")
    base_model.eval()
    PN = ProjNorm(base_model=base_model)

    if args['use_base_model']:
        print('using base model...')
        iid_model = 'base'
        PN.id_model = base_model
    else:
        print('using ref model...')
        iid_model = 'ref'
        ref_model = torch.load(f"{save_dir_path}/ref_model_{args['model_seed'].split('_')[0]}_{args['pseudo_iters']}.pt")
        ref_model.eval()
        PN.id_model = ref_model

    ################ train ood pseudo model ################
    if args['arch'] == 'resnet18':
        pseudo_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'resnet50':
        pseudo_model = ResNet50(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'vgg11':
        pseudo_model = VGG11(num_classes=args['num_classes'], seed=args['seed']).cuda()
    else:
        raise ValueError('incorrect model name')

    PN.update_pseudo_model(val_ood_loader,
                           pseudo_model,
                           lr=args['lr'],
                           pseudo_iters=args['pseudo_iters'])

    # compute OOD ProjNorm
    projnorm_value = PN.compute_projnorm(PN.id_model, PN.pseudo_model)

    print('===========out-of-distribution===========')
    print('(out-of-distribution) ProjNorm value: ', projnorm_value)
    test_loss_ood, test_error_ood = evaluation(net=base_model, testloader=val_ood_loader)
    print('(out-of-distribution) test error: ', test_error_ood)
    print('========finished========')

    dataset = os.path.basename(args['data_path'])
    corruption = args['corruption']
    result_dir = f"results/{dataset}/{args['arch']}_{args['model_seed']}/ProjNorm-{iid_model}_{n_ood_sample}/{corruption}.json"
    print(result_dir, os.path.dirname(result_dir), os.path.basename(result_dir))
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    if not os.path.exists(result_dir):
        with open(result_dir, 'w') as f:
            json.dump([], f)

    with open(result_dir, 'r') as f:
        data = json.load(f)
    
    data.append({
        'corruption': corruption,
        'corruption level': args['severity'],
        'method': 'ProjNorm',
        'metric': projnorm_value,
        'acc': test_error_ood / 100,
        'error': 1 - test_error_ood / 100
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)


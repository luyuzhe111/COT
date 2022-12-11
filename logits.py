import argparse
from projnorm import *
from load_data import *
from model import ResNet18, ResNet50, VGG11
from utils import evaluation, gather_outputs
import json
import torch
import time
from netdissect import nethook
from tqdm import tqdm
import ot

"""# Configuration"""
parser = argparse.ArgumentParser(description='UWD.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--layer', default='fc', type=str)
parser.add_argument('--pooling', default='logits', type=str)
parser.add_argument('--metric', default='wd', type=str)
parser.add_argument('--cifar_data_path', default='./data/CIFAR-10', type=str)
parser.add_argument('--cifar_corruption_path', default='./data/CIFAR-10-C/numpy_format', type=str)
parser.add_argument('--corruption', default='snow', type=str)
parser.add_argument('--severity', default=1, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_ood_samples', default=10000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_base_model', action='store_true',
                    default=False, help='apply base_model for computing ProjNorm')
args = vars(parser.parse_args())

print(args)

if __name__ == "__main__":
    # setup valset_iid/val_ood loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args['seed'])
    random_seeds = torch.randint(0, 10000, (2,))
    
    type = "cifar-100" if args['num_classes'] == 100 else "cifar-10"

    n_ood_sample = args['num_ood_samples']

    valset_iid = load_cifar_image(corruption_type='clean',
                                  clean_cifar_path=args['cifar_data_path'],
                                  corruption_cifar_path=args['cifar_corruption_path'],
                                  corruption_severity=0,
                                  datatype='test',
                                  type=type,
                                  seed=random_seeds[0])
    
    val_iid_loader = torch.utils.data.DataLoader(valset_iid,
                                                 batch_size=args['batch_size'],
                                                 shuffle=True)

    valset_ood = load_cifar_image(corruption_type=args['corruption'],
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype='test',
                                    num_samples=n_ood_sample,
                                    type=type,
                                    seed=random_seeds[1])
    
    val_ood_loader = torch.utils.data.DataLoader(valset_ood,
                                                 batch_size=args['batch_size'],
                                                 shuffle=True)
    
    cache_dir = f"./cache/{type}/{args['arch']}-{args['layer']}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_id_dir = f"{cache_dir}/id_{random_seeds[0]}.pkl"
    cache_od_dir = f"{cache_dir}/od_{args['corruption']}_n{n_ood_sample}_{args['severity']}_{random_seeds[1]}.pkl"

    # init ProjNorm
    save_dir_path = f"./checkpoints/{type}/{args['arch']}"

    base_model = torch.load('{}/base_model.pt'.format(save_dir_path), map_location=device)
    model = nethook.InstrumentedModel(base_model).eval().to(device)
    model.eval()

    layer = args['layer']
        
    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, args['pooling'], layer, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, args['pooling'], layer, device, cache_od_dir)

    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    print('in-distribution acc:', iid_acc)
    print('out-distribution acc:', ood_acc)

    def compute_swd(p, q):
        wasserstein_distance = torch.pow( (torch.sort(p, dim=1)[0] - torch.sort(q, dim=1)[0]), 2 )

        return wasserstein_distance.mean().item()

    metric = args['metric']

    if metric == 'fwd':
        dist = compute_swd(iid_acts, ood_acts)
    elif metric == 'swd':
        n_class = args['num_classes']
        proj = torch.as_tensor(ot.sliced.get_random_projections(n_class, n_class * 10, seed=0)).to(device=device, dtype=torch.float)
        dist = ot.sliced.sliced_wasserstein_distance(iid_acts, ood_acts, projections=proj)
    elif metric == 'wd':
        M = ot.dist(iid_acts, ood_acts)
        weights = torch.as_tensor([]).to(device)
        dist = ot.emd2(weights, weights, M, numItermax=10**8)

    print(f'{metric} distance:', dist.item())

    dataset = os.path.basename(args['cifar_data_path'])
    corruption = args['corruption']
    result_dir = f"results/{dataset}/{args['arch']}/{args['metric']}_{n_ood_sample}/{args['pooling']}/{layer}/{corruption}.json"
    print(result_dir, os.path.dirname(result_dir), os.path.basename(result_dir))
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    if not os.path.exists(result_dir):
        with open(result_dir, 'w') as f:
            data = [
                {
                    'corruption': corruption,
                    'corruption level': 0,
                    'swd': 0,
                    'acc': float(iid_acc),
                    'error': 1 - iid_acc
                }
            ]
            json.dump(data, f)

    with open(result_dir, 'r') as f:
        data = json.load(f)
    
    data.append({
        'corruption': corruption,
        'corruption level': args['severity'],
        'swd': float(dist),
        'acc': float(ood_acc),
        'error': 1 - ood_acc
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)


import argparse
from projnorm import *
from load_data import *
from model import ResNet18, ResNet50, VGG11
from misc.temperature_scaling import calibrate
from collections import Counter
from utils import gather_outputs
from misc.torch_interp import interpolate
import numpy as np
import json
import torch
import math
import os
from tqdm import tqdm
import ot
import ot.dr
import torch.nn as nn


def main():
    """# Configuration"""
    parser = argparse.ArgumentParser(description='UWD.')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--metric', default='mini-wd', type=str)
    parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
    parser.add_argument('--corruption_path', default='./data/CIFAR-10-C/numpy_format', type=str)
    parser.add_argument('--data_type', default='cifar-10', type=str)
    parser.add_argument('--corruption', default='snow', type=str)
    parser.add_argument('--severity', default=1, type=int)
    parser.add_argument('--ref', default='val', type=str)
    parser.add_argument('--num_ref_samples', default=50000, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_ood_samples', default=10000, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--reg', default=100, type=int)
    parser.add_argument('--model_seed', default="1", type=str)
    parser.add_argument('--seed', default=1, type=int)
    args = vars(parser.parse_args())

    print(args)

    # setup valset_iid/val_ood loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args['seed'])

    model_seed = args['model_seed']

    n_ood_sample = args['num_ood_samples']
    n_ref_sample = args['num_ref_samples']
    n_class = args['num_classes']

    data_type = args['data_type']

    if args['ref'] == 'val':
        _, val_set = load_image_dataset(corruption_type='clean',
                                        clean_path=args['data_path'],
                                        corruption_path=args['corruption_path'],
                                        corruption_severity=0,
                                        num_samples=n_ref_sample,
                                        datatype='train',
                                        type=data_type,
                                        seed=args['seed']
                                        )
    test_set = load_image_dataset(corruption_type='clean',
                                    clean_path=args['data_path'],
                                    corruption_path=args['corruption_path'],
                                    corruption_severity=0,
                                    num_samples=args['num_ood_samples'],
                                    datatype='test',
                                    type=data_type,
                                    seed=args['seed']
                                    )
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['batch_size'], shuffle=False)

    val_iid_loader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'], shuffle=False)

    valset_ood = load_image_dataset(corruption_type=args['corruption'],
                                    clean_path=args['data_path'],
                                    corruption_path=args['corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype='test',
                                    num_samples=n_ood_sample,
                                    type=data_type
                                    )

    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=128, shuffle=True)

    cache_dir = f"./cache/{data_type}/{args['arch']}_{model_seed}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_id_dir = f"{cache_dir}/id_{model_seed}_{args['ref']}_d{args['seed']}.pkl"
    cache_od_dir = f"{cache_dir}/od_{model_seed}_{args['corruption']}-{args['severity']}_n{n_ood_sample}.pkl"

    save_dir_path = f"./checkpoints/{data_type}/{args['arch']}"

    base_model = torch.load(f"{save_dir_path}/base_model_{args['model_seed']}.pt", map_location=device)
    model = base_model.eval()

    temp_dir = f"{save_dir_path}/base_model_{args['model_seed']}_temp.json"
    model = calibrate(model, val_iid_loader, temp_dir)

    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, device, cache_od_dir)

    print('iid labels: ', Counter(iid_tars.tolist()))
    print('ood labels: ', Counter(ood_tars.tolist()))

    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    print('validation acc:', iid_acc)
    print('out-distribution acc:', ood_acc)

    metric = args['metric']
    match_rate = 0

    if metric == 'mmd':
        xx, yy, zz = torch.mm(iid_acts, iid_acts.t()), torch.mm(ood_acts, ood_acts.t()), torch.mm(iid_acts, ood_acts.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        a = 100
        XX = torch.exp(-0.5*dxx/a)
        YY = torch.exp(-0.5*dyy/a)
        XY = torch.exp(-0.5*dxy/a)

        dist = torch.mean(XX + YY - 2. * XY)  

    elif metric == 'EMD':
        act = nn.Softmax(dim=1)
        
        # n_class_sample = len(iid_tars) // n_class
        # ref_tars = torch.as_tensor(list(range(n_class)) * n_class_sample)
        # ref_acts = nn.functional.one_hot(ref_tars)

        # iid_acts, ood_acts = ref_acts, act(ood_acts)
        iid_acts, ood_acts = nn.functional.one_hot(iid_tars), act(ood_acts)
        
        M = torch.from_numpy(ot.dist(iid_acts.cpu().numpy(), ood_acts.cpu().numpy(), metric='minkowski', p=1)).to(device)
        
        weights = torch.as_tensor([]).to(device)
        G0 = ot.emd(weights, weights, M, numItermax=10**8)
        source_iid_inds = G0.nonzero()[:, 0]
        matched_ood_inds = G0.nonzero()[:, 1]
        
        match_rate = (iid_tars == ood_preds[matched_ood_inds]).float().mean().item()

        dist = M[source_iid_inds, matched_ood_inds].mean() / 2

    elif metric == 'sinkhorn':
        dist = ot.bregman.empirical_sinkhorn2(iid_acts, ood_acts, reg=args['reg'])

    elif metric == 'swd':
        n_class = args['num_classes']
        n_slice = n_class * 100
        proj = torch.as_tensor(ot.sliced.get_random_projections(n_class, n_slice, seed=0)).to(device=device, dtype=torch.float)

        slice_batch_size = 1000
        n_batch = math.ceil(n_slice // slice_batch_size)
        dist = 0
        for i in range(n_batch):
            proj_batch = proj[:, i*slice_batch_size: (i+1)*slice_batch_size]
            iid_acts_batch = iid_acts @ proj_batch
            ood_acts_batch = ood_acts @ proj_batch
            iid_sorted_batch = torch.sort(iid_acts_batch, dim=0)[0]
            ood_sorted_batch = torch.sort(ood_acts_batch, dim=0)[0]

            p_interp, q_interp = interpolate(iid_sorted_batch, ood_sorted_batch)
            dist += torch.pow( (p_interp - q_interp), 2 ).sum(0).sum()

        dist /= n_slice

    elif metric == 'cwd':
        p = torch.sort(iid_acts, dim=0)[0]
        q = torch.sort(ood_acts, dim=0)[0]

        p_interp, q_interp = interpolate(p, q)

        dist = torch.pow( p_interp - q_interp, 2).sum(0).mean()

    print(f'{metric} distance:', dist.item())
    print(f'matching rate:', match_rate)

    dataset = os.path.basename(args['data_path'])
    corruption = args['corruption']

    result_dir = f"results/{dataset}/{args['arch']}_{model_seed}/{args['metric']}_{args['ref']}_{n_ood_sample}/{corruption}.json"

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
        'metric': float(dist),
        'ref': args['metric'],
        'acc': float(ood_acc),
        'error': 1 - ood_acc,
        'match_rate': match_rate,
        'mismatch_rate': 1 - match_rate
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()


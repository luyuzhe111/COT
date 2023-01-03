import argparse
from projnorm import *
from load_data import *
from model import ResNet18, ResNet50, VGG11
from utils import gather_outputs, interpolate
import numpy as np
import json
import torch
import math
import os
from tqdm import tqdm
from sklearn.decomposition import PCA, KernelPCA
import ot
import ot.dr
import torch.nn as nn

def main():
    """# Configuration"""
    parser = argparse.ArgumentParser(description='UWD.')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--metric', default='mini-wd', type=str)
    parser.add_argument('--cifar_data_path', default='./data/CIFAR-10', type=str)
    parser.add_argument('--cifar_corruption_path', default='./data/CIFAR-10-C/numpy_format', type=str)
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
    random_seeds = torch.randint(0, 10000, (2,))

    model_seed = args['model_seed']

    n_ood_sample = args['num_ood_samples']
    n_ref_sample = args['num_ref_samples']
    
    type = "cifar-100" if args['num_classes'] == 100 else "cifar-10"

    if args['ref'] == 'val':
        _, val_set = load_cifar_image(corruption_type='clean',
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=0,
                                    num_samples=n_ref_sample,
                                    datatype='train',
                                    type=type,
                                    seed=random_seeds[0]
                                    )
    else:
        val_set = load_cifar_image(corruption_type='clean',
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=0,
                                    num_samples=10000,
                                    datatype='test',
                                    type=type,
                                    seed=random_seeds[0]
                                    )

    
    val_iid_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)

    valset_ood = load_cifar_image(corruption_type=args['corruption'],
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype='test',
                                    num_samples=n_ood_sample,
                                    type=type,
                                    seed=random_seeds[1])
    
    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=128, shuffle=True)
    
    cache_dir = f"./cache/{type}/{args['arch']}_{model_seed}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_id_dir = f"{cache_dir}/id_{args['ref']}_{model_seed}.pkl"
    cache_od_dir = f"{cache_dir}/od_{model_seed}_{args['corruption']}_n{n_ood_sample}_{args['severity']}_{random_seeds[1]}.pkl"

    save_dir_path = f"./checkpoints/{type}/{args['arch']}"

    base_model = torch.load(f"{save_dir_path}/base_model_{args['model_seed']}.pt", map_location=device)
    model = base_model.eval()
        
    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, device, cache_od_dir)

    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    print('validation acc:', iid_acc)
    print('out-distribution acc:', ood_acc)

    metric = args['metric']

    if metric == 'mmd':
        iid_mean = iid_acts.mean(0)
        ood_mean = ood_acts.mean(0)

        dist = ( (iid_mean - ood_mean) ** 2 ).mean()
    
    elif metric == 'sinkhorn':
        dist = ot.bregman.empirical_sinkhorn2(iid_acts, ood_acts, reg=args['reg'])

    elif metric == 'wd':
        M = ot.dist(iid_acts, ood_acts)
        weights = torch.as_tensor([]).to(device)
        dist = ot.emd2(weights, weights, M, numItermax=10**8)
    
    elif metric == 'smwd':
        act = nn.Softmax(dim=1)
        sm_iid_acts, sm_ood_acts = act(iid_acts), act(ood_acts)
        M = ot.dist(sm_iid_acts, sm_ood_acts)
        weights = torch.as_tensor([]).to(device)
        dist = ot.emd2(weights, weights, M, numItermax=10**8)
    
    elif metric == 'swd':
        n_class = args['num_classes']
        n_slice = n_class * 100
        proj = torch.as_tensor(ot.sliced.get_random_projections(n_class, n_slice, seed=0)).to(device=device, dtype=torch.float)
        
        slice_batch_size = 100
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
    
    elif metric == 'ncwd':
        iid_acts_min = torch.amin(iid_acts, 0)
        iid_acts_max = torch.amax(iid_acts, 0)

        n_iid_acts = (iid_acts - iid_acts_min) / ( iid_acts_max - iid_acts_min )
        n_ood_acts = (ood_acts - iid_acts_min) / ( iid_acts_max - iid_acts_min )

        p = torch.sort(n_iid_acts, dim=0)[0]
        q = torch.sort(n_ood_acts, dim=0)[0]

        p_interp, q_interp = interpolate(p, q)
        
        dist = torch.pow( p_interp - q_interp, 2).sum(0).mean()
    
    elif metric == 'smncwd':
        act = nn.Softmax(dim=1)
        iid_acts, ood_acts = act(iid_acts), act(ood_acts)

        iid_acts_min = torch.amin(iid_acts, 0)
        iid_acts_max = torch.amax(iid_acts, 0)

        n_iid_acts = (iid_acts - iid_acts_min) / ( iid_acts_max - iid_acts_min )
        n_ood_acts = (ood_acts - iid_acts_min) / ( iid_acts_max - iid_acts_min )

        p = torch.sort(n_iid_acts, dim=0)[0]
        q = torch.sort(n_ood_acts, dim=0)[0]

        p_interp, q_interp = interpolate(p, q)
        
        dist = torch.pow( p_interp - q_interp, 2).sum(0).mean()

    elif metric == 'smcwd':
        act = nn.Softmax(dim=1)
        sm_iid_acts, sm_ood_acts = act(iid_acts), act(ood_acts)

        p = torch.sort(sm_iid_acts, dim=0)[0]
        q = torch.sort(sm_ood_acts, dim=0)[0]

        p_interp, q_interp = interpolate(p, q)
        
        dist = torch.pow( p_interp - q_interp, 2).sum(0).mean()
    
    elif metric == 'mini-wd':
        torch.manual_seed(0)
        sub_inds = torch.randperm(args['batch_size'])
        sub_iid_acts = iid_acts[sub_inds]
        
        ood_size = len(ood_acts)
        batch_size = args['batch_size']
        n_batch = math.ceil(ood_size // batch_size)
        perm_inds = torch.randperm(ood_size)
        perm_ood_acts = ood_acts[perm_inds]

        dist = 0
        for i in range(n_batch):
            print(f'wd for sample {i * batch_size} - {(i+1) * batch_size}')
            ood_acts_batch = perm_ood_acts[i * batch_size: (i+1) * batch_size]
            
            # batch wd
            M = ot.dist(sub_iid_acts, ood_acts_batch)
            weights = torch.as_tensor([]).to(device)
            dist += ot.emd2(weights, weights, M, numItermax=10**7)
        
        dist /= n_batch

    print(f'{metric} distance:', dist.item())

    dataset = os.path.basename(args['cifar_data_path'])
    corruption = args['corruption']

    if 'mini' in metric:
        result_dir = f"results/{dataset}/{args['arch']}_{model_seed}/{args['metric']}_{args['ref']}_b{batch_size}_{n_ood_sample}/{corruption}.json"
    else:
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
        'error': 1 - ood_acc
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()


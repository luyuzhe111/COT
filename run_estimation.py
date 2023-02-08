import argparse
from projnorm import *
from load_data import load_train_dataset, load_test_dataset
from model import ResNet18, ResNet50, VGG11
from misc.temperature_scaling import calibrate
from collections import Counter
from utils import gather_outputs
from misc.torch_interp import interpolate
from torch_datasets.configs import get_expected_label_distribution
import numpy as np
import json
import torch
from utils import compute_t
import os
from tqdm import tqdm
import ot
import ot.dr
import torch.nn as nn


def main():
    # generic configs
    parser = argparse.ArgumentParser(description='Estimate target domain performance.')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--metric', default='EMD', type=str)
    parser.add_argument('--dataset', default='cifar-10', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_val_samples', default=10000, type=int)
    parser.add_argument('--n_test_samples', default=1000, type=int)
    parser.add_argument('--dataset_seed', default=1, type=int)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--ckpt_epoch', default=20, type=int)

    # synthetic shifts configs
    parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
    parser.add_argument('--corruption_path', default='./data/CIFAR-10-C/', type=str)
    parser.add_argument('--corruption', default='brightness', type=str)
    parser.add_argument('--severity', default=1, type=int)
    
    args = parser.parse_args()

    print(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_seed = args.model_seed
    n_test_sample = args.n_test_samples
    dsname = args.dataset
    corruption = args.corruption
    severity = args.severity

    # load in iid data for calibration
    _, val_set = load_train_dataset(dsname=dsname,
                                    iid_path=args.data_path,
                                    n_val_samples=args.n_val_samples,
                                    seed=args.dataset_seed)

    val_iid_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # load in ood test data 
    valset_ood = load_test_dataset(dsname=dsname,
                                   iid_path=args.data_path,
                                   corr_path=args.corruption_path,
                                   corr_type=args.corruption,
                                   corr_sev=args.severity,
                                   n_test_sample=n_test_sample)

    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=args.batch_size, shuffle=True)

    cache_dir = f"./cache/{dsname}/{args.arch}_{model_seed}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_id_dir = f"{cache_dir}/id_{model_seed}_d{args.dataset_seed}.pkl"
    cache_od_dir = f"{cache_dir}/od_{model_seed}_{args.corruption}-{args.severity}_n{n_test_sample}.pkl"

    save_dir_path = f"./checkpoints/{dsname}/{args.arch}"

    base_model = torch.load(f"{save_dir_path}/base_model_{args.model_seed}.pt", map_location=device)
    model = base_model.eval()
    
    # use temperature scaling to calibrate model
    temp_dir = f"{save_dir_path}/base_model_{args.model_seed}_temp.json"
    model = calibrate(model, val_iid_loader, temp_dir)

    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, device, cache_od_dir)

    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    conf = torch.nn.functional.softmax(iid_acts, dim=1).amax(1).mean().item()
    conf_gap =  conf - iid_acc

    print('validation acc:', iid_acc)
    print('validation confidence:', conf)
    print('confidence gap:', conf - iid_acc)
    print('out-distribution acc:', ood_acc)

    metric = args.metric

    if metric == 'EMD':
        act = nn.Softmax(dim=1)

        iid_acts, ood_acts = nn.functional.one_hot(iid_tars), act(ood_acts)
        
        M = torch.from_numpy(ot.dist(iid_acts.cpu().numpy(), ood_acts.cpu().numpy(), metric='minkowski', p=1)).to(device)
        weights = torch.as_tensor([]).to(device)
        est = ( ot.emd2(weights, weights, M, numItermax=10**8) / 2 + conf_gap ).item()
    
    elif metric == 'REMD':
        act = nn.Softmax(dim=1)
        iid_acts, ood_acts = nn.functional.one_hot(iid_tars), act(ood_acts)
        reduction_rate = int(metric.split('_')[-1])

        def reduce_classes(acts):
            cur_n_class = acts.shape[1]
            tar_n_class = cur_n_class // reduction_rate
            if cur_n_class < tar_n_class:
                return acts
            else:
                chunk_size = cur_n_class // tar_n_class
                chunks = []
                for i in range(tar_n_class):
                    chunks.append(
                        acts[:, i * chunk_size : (i+1) * chunk_size].sum(1).unsqueeze(1)
                    )
                return torch.cat(chunks, dim=1)

        riid_acts = reduce_classes(iid_acts)
        rood_acts = reduce_classes(ood_acts)
        
        M = torch.from_numpy(ot.dist(riid_acts.cpu().numpy(), rood_acts.cpu().numpy(), metric='minkowski', p=1)).to(device)
        weights = torch.as_tensor([]).to(device)
        est = ( ot.emd2(weights, weights, M, numItermax=10**8) / 2 + conf_gap ).item()
    
    elif metric == 'ATC':
        cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}/iid_result.json"
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                t = data['t']
        else:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            with open(cache_dir, 'w') as f:
                print('compute confidence threshold...')
                t = compute_t(model, val_iid_loader).item()
                json.dump({'t': t}, f)
        
        softmax = nn.Softmax(dim=1)
        s_softmax = torch.sum(softmax(ood_acts) * torch.log2(softmax(ood_acts)), dim=1)
        est = (s_softmax < t).sum().item() / len(ood_acts)

    print(f'{metric} value:', est)

    result_dir = f"results/{dsname}/{args.arch}_{model_seed}/{args.metric}_{n_test_sample}/{corruption}.json"

    print(result_dir, os.path.dirname(result_dir), os.path.basename(result_dir))
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    if not os.path.exists(result_dir):
        with open(result_dir, 'w') as f:
            json.dump([], f)

    with open(result_dir, 'r') as f:
        data = json.load(f)
    
    data.append({
        'corruption': corruption,
        'corruption level': severity,
        'metric': float(est),
        'ref': metric,
        'acc': float(ood_acc),
        'error': 1 - ood_acc,
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()


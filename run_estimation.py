import argparse
from load_data import load_train_dataset, load_test_dataset
from model import ResNet18, ResNet50, VGG11
from misc.temperature_scaling import calibrate
from collections import Counter
from utils import gather_outputs, compute_t, compute_cott
from misc.torch_interp import interpolate
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch_datasets.configs import get_expected_label_distribution, get_n_classes
from tqdm import tqdm
import ot



def main():
    # generic configs
    parser = argparse.ArgumentParser(description='Estimate target domain performance.')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--metric', default='EMD', type=str)
    parser.add_argument('--dataset', default='cifar-10', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_val_samples', default=10000, type=int)
    parser.add_argument('--n_test_samples', default=10000, type=int)
    parser.add_argument('--dataset_seed', default=1, type=int)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--ckpt_epoch', default=20, type=int)

    # synthetic shifts configs
    parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
    parser.add_argument('--subpopulation', default='same', type=str)
    parser.add_argument('--corruption_path', default='./data/CIFAR-10-C/', type=str)
    parser.add_argument('--corruption', default='clean', type=str)
    parser.add_argument('--severity', default=0, type=int)
    
    args = parser.parse_args()

    print(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = args.metric
    pretrained = args.pretrained
    model_seed = args.model_seed
    model_epoch = args.ckpt_epoch
    n_test_sample = args.n_test_samples
    dsname = args.dataset
    corruption = args.corruption
    severity = args.severity
    n_class = get_n_classes(args.dataset)

    # load in iid data for calibration
    _, val_set = load_train_dataset(dsname=dsname,
                                    iid_path=args.data_path,
                                    pretrained=pretrained,
                                    n_val_samples=args.n_val_samples,
                                    seed=args.dataset_seed)

    val_iid_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # load in ood test data 
    valset_ood = load_test_dataset(dsname=dsname,
                                   subpopulation=args.subpopulation,
                                   iid_path=args.data_path,
                                   corr_path=args.corruption_path,
                                   corr=args.corruption,
                                   corr_sev=args.severity,
                                   pretrained=pretrained,
                                   n_test_sample=n_test_sample)

    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=args.batch_size, shuffle=True, num_workers=4)

    n_test_sample = len(valset_ood)

    if pretrained:
        cache_dir = f"./cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/pretrained"
    else:
        cache_dir = f"./cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/scratch"

    os.makedirs(cache_dir, exist_ok=True)
    cache_id_dir = f"{cache_dir}/id_m{model_seed}-{model_epoch}_d{args.dataset_seed}.pkl"
    cache_od_dir = f"{cache_dir}/od_p{args.subpopulation}_m{model_seed}-{model_epoch}_c{corruption}-{severity}_n{n_test_sample}.pkl"

    if pretrained:
        save_dir_path = f"./checkpoints/{dsname}/{args.arch}/pretrained"
    else:
        save_dir_path = f"./checkpoints/{dsname}/{args.arch}/scratch"

    ckpt = torch.load(f"{save_dir_path}/base_model_{args.model_seed}-{model_epoch}.pt", map_location=device)
    model = ckpt['model']
    model.eval()
    
    # use temperature scaling to calibrate model
    print('calibrating models...')
    
    opt_bias = True
    if opt_bias:
        temp_dir = f"{cache_dir}/base_model_{args.model_seed}-{model_epoch}_temp_with_bias.json"
    else:
        temp_dir = f"{cache_dir}/base_model_{args.model_seed}-{model_epoch}_temp_.json"
    model = calibrate(model, n_class, opt_bias, val_iid_loader, temp_dir)
    print('calibration done.')

    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, device, cache_od_dir)

    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    conf = torch.nn.functional.softmax(iid_acts, dim=1).amax(1).mean().item()
    conf_gap =  conf - iid_acc

    print('n ood test sample:', n_test_sample)

    print('------------------')
    print(f'validation acc:', iid_acc)
    print('validation confidence:', conf)
    print('confidence gap:', conf - iid_acc)
    print('------------------')
    print()

    ood_preds_count = Counter(ood_preds.tolist())
    ood_tars_count = Counter(ood_tars.tolist())

    iid_tars_dist = get_expected_label_distribution(args.dataset)
    ood_tars_dist = [ood_tars_count[i] / len(ood_acts) for i in range(n_class)]
    ood_preds_dist = [ood_preds_count[i] / len(ood_acts) for i in range(n_class)]

    print('------------------')
    print("ood real label tv:", sum(abs(np.array(ood_tars_dist) - np.array(iid_tars_dist))) / 2 )
    print("ood pseudo label tv:", sum(abs(np.array(ood_preds_dist) - np.array(iid_tars_dist))) / 2 )
    print("ood pseudo-real label tv:", sum(abs(np.array(ood_preds_dist) - np.array(ood_tars_dist))) / 2 )
    print('------------------')
    print()

    if metric == 'COT':
        exp_labels = torch.as_tensor(random.choices(
            list(range(n_class)), weights=get_expected_label_distribution(args.dataset), k=n_test_sample
        ))
        iid_acts = nn.functional.one_hot(exp_labels)
        ood_acts = nn.functional.softmax(ood_acts, dim=-1)
        M = torch.sum( torch.abs( iid_acts.unsqueeze(1) - ood_acts.unsqueeze(0) ), dim=-1 ) / 2
        weights = torch.as_tensor([])
        est = ( ot.emd2(weights, weights, M, numItermax=10**8) / 2 + conf_gap ).item()
    
    elif metric == 'COTT':
        if pretrained:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/pretrained_cott_threshold.json"
        else:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/scratch_cott_threshold.json"
        
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                thresholds = data['t']
        else:
            with open(cache_dir, 'w') as f:
                print('compute confidence threshold...')
                thresholds = compute_cott(model, val_iid_loader, n_class)
                json.dump({'t': thresholds.tolist()}, f)

        exp_labels = torch.as_tensor( random.choices(
                list(range(n_class)), weights=get_expected_label_distribution(args.dataset), k=n_test_sample
        ) )
        iid_acts = nn.functional.one_hot(exp_labels)
        ood_acts = nn.functional.softmax(ood_acts, dim=-1).cpu()

        M = torch.sum( torch.abs( iid_acts.unsqueeze(1) - ood_acts.unsqueeze(0) ), dim=-1 )
        weights = torch.as_tensor([])
        Pi = ot.emd(weights, weights, M, numItermax=10**8)
        
        label_inds = Pi.nonzero()[:, 0]
        matched_softmax_inds = Pi.nonzero()[:, 1]

        est = 0
        for i in range(n_class):
            clss_tar_inds = ( torch.argmax(iid_acts, dim=1) == i )
            clss_scores = M[ label_inds[clss_tar_inds], matched_softmax_inds[clss_tar_inds] ]
            clss_est = ( clss_scores > thresholds[i] ).sum()
            est += clss_est
        
        est = est.item() / n_test_sample
    
    elif metric == 'ATC':
        if pretrained:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/pretrained_atc_threshold.json"
        else:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/scratch_atc_threshold.json"
        
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

    print('------------------')
    print('True OOD error:', 1 - ood_acc)
    print(f'{metric} predicted OOD error:', est)
    print('------------------')
    print()

    n_test_str = n_test_sample if dsname not in ['FMoW'] else -1
    if pretrained:
        result_dir = f"results/{dsname}/pretrained/{args.arch}_{model_seed}-{model_epoch}/{args.metric}_{n_test_str}/{corruption}.json"
    else:
        result_dir = f"results/{dsname}/scratch/{args.arch}_{model_seed}-{model_epoch}/{args.metric}_{n_test_str}/{corruption}.json"

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
        'subpopulation': args.subpopulation,
        'pretrained': pretrained
    })

    with open(result_dir, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()


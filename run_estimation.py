import argparse
from load_data import load_val_dataset, load_test_dataset
from model import ResNet18, ResNet50, VGG11
from misc.temperature_scaling import calibrate
from collections import Counter
from utils import gather_outputs, get_threshold, get_im_estimate, get_temp_dir
from misc.torch_interp import interpolate
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch_datasets.configs import get_n_classes, get_expected_label_distribution, sample_label_dist
from tqdm import tqdm
import time
import math
import ot



def main():
    # generic configs
    parser = argparse.ArgumentParser(description='Estimate target domain performance.')
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--metric', default='EMD', type=str)
    parser.add_argument('--dataset', default='cifar-10', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_val_samples', default=10000, type=int)
    parser.add_argument('--n_test_samples', default=-1, type=int)
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
    val_set = load_val_dataset(dsname=dsname,
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
    
    val_ood_loader = torch.utils.data.DataLoader(valset_ood, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
    
    opt_bias = False
    temp_dir = get_temp_dir(cache_dir, model_seed, model_epoch, opt_bias=opt_bias)
    
    model = calibrate(model, n_class, opt_bias, val_iid_loader, temp_dir)
    print('calibration done.')

    iid_acts, iid_preds, iid_tars = gather_outputs(model, val_iid_loader, device, cache_id_dir)
    ood_acts, ood_preds, ood_tars = gather_outputs(model, val_ood_loader, device, cache_od_dir)
    
    act_fn = nn.Softmax(dim=1)
    iid_acts = act_fn(iid_acts).cpu()
    ood_acts = act_fn(ood_acts).cpu()
    
    iid_acc = ( (iid_preds == iid_tars).sum() / len(iid_tars) ).item()
    ood_acc = ( (ood_preds == ood_tars).sum() / len(ood_tars) ).item()

    conf = iid_acts.amax(1).mean().item()

    print('n ood test sample:', n_test_sample)

    print('------------------')
    print('validation acc:', iid_acc)
    print('validation confidence:', conf)
    print('confidence gap:', conf - iid_acc)
    print('------------------')
    print()

    ood_preds_count = Counter(ood_preds.tolist())
    ood_tars_count = Counter(ood_tars.tolist())
    iid_preds_count = Counter(iid_tars.tolist())

    iid_tars_dist = get_expected_label_distribution(args.dataset)
    ood_tars_dist = [ood_tars_count[i] / len(ood_acts) for i in range(n_class)]
    ood_preds_dist = [ood_preds_count[i] / len(ood_acts) for i in range(n_class)]
    iid_preds_dist = [iid_preds_count[i] / len(iid_acts) for i in range(n_class)]

    print('------------------')
    print("ood real label tv:", sum(abs(np.array(ood_tars_dist) - np.array(iid_tars_dist))) / 2 )
    print("ood pseudo label tv:", sum(abs(np.array(ood_preds_dist) - np.array(iid_preds_dist))) / 2 )
    print("ood pseudo-real label tv:", sum(abs(np.array(ood_preds_dist) - np.array(ood_tars_dist))) / 2 )
    print('------------------')
    print()
    
    start = time.time()
    
    if metric == 'AC':
        max_confidence = torch.max(ood_acts, dim=-1)[0]
        est = 1 - torch.mean(max_confidence).item()
    
    elif metric == 'DoC':
        source_prob = iid_acts.max(1)[0]
        target_prob = ood_acts.max(1)[0]
        source_err = (iid_preds != iid_tars).sum().item() / len(iid_tars)
        est = source_err +  torch.mean(source_prob).item() - torch.mean(target_prob).item()
    
    elif metric == 'IM':
        source_prob = iid_acts.max(1)[0]
        target_prob = ood_acts.max(1)[0]
        est = get_im_estimate(source_prob, target_prob, (iid_preds == iid_tars).cpu()).item()
    
    elif metric == 'GDE':
        seeds = [0, 1, 10]
        seed_ind = seeds.index(model_seed)
        alt_model_seed = seeds[ (seed_ind + 1) % len(seeds) ]
        alt_ckpt = torch.load(f"{save_dir_path}/base_model_{alt_model_seed}-{model_epoch}.pt", map_location=device)
        alt_model = alt_ckpt['model']
        alt_model.eval()
        
        if pretrained:
            alt_cache_dir = f"./cache/{dsname}/{args.arch}_{alt_model_seed}-{model_epoch}/pretrained"
        else:
            alt_cache_dir = f"./cache/{dsname}/{args.arch}_{alt_model_seed}-{model_epoch}/scratch"
        
        os.makedirs(alt_cache_dir, exist_ok=True)
        
        alt_temp_dir = get_temp_dir(alt_cache_dir, alt_model_seed, model_epoch, opt_bias=opt_bias)
        alt_model = calibrate(alt_model, n_class, opt_bias, val_iid_loader, alt_temp_dir)
        
        alt_cache_od_dir = f"{alt_cache_dir}/od_p{args.subpopulation}_m{alt_model_seed}-{model_epoch}_c{corruption}-{severity}_n{n_test_sample}.pkl"
        _, alt_ood_preds, _ = gather_outputs(alt_model, val_ood_loader, device, alt_cache_od_dir)
        
        est = alt_ood_preds.ne(ood_preds).sum().item() / len(alt_ood_preds)
    
    elif metric == 'ATC-MC':
        threshold = get_threshold(model, val_iid_loader, n_class, args)
        mc = ood_acts.max(1)[0]
        est = (mc < threshold).sum().item() / len(ood_acts)
        cost_dist = torch.sort(mc)[0].tolist()
        
    elif metric == 'ATC-NE':
        threshold = get_threshold(model, val_iid_loader, n_class, args)
        ne = torch.sum(ood_acts * torch.log2(ood_acts), dim=1)
        est = (ne < threshold).sum().item() / len(ood_acts)
        cost_dist = torch.sort(ne)[0].tolist()
    
    elif metric == 'Pseudo':
        est = min(
            sum(abs(np.array(ood_preds_dist) - np.array(iid_preds_dist))) / 2 + (1 - iid_acc), 1
        )

    elif metric == 'COT':
        batch_size = min(10000, n_test_sample)
        n_batch = math.ceil( n_test_sample // batch_size)
        
        print(
            f'total of {n_test_sample} test samples, running {n_batch} batches.'
        )
        
        if n_batch > 1:
            est = 0
            random.seed(0)
            for _ in range(n_batch):
                rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                iid_acts_batch = nn.functional.one_hot(
                    sample_label_dist(dsname, n_class, batch_size)
                )
                ood_acts_batch = ood_acts[rand_inds]
                
                M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                weights = torch.as_tensor([])
                est += ( ot.emd2(weights, weights, M, numItermax=1e8, numThreads=8) / 2 ).item()
            est = est / n_batch
        else:
            torch.manual_seed(0)
            exp_labels = sample_label_dist(dsname, n_class, len(ood_acts))
            iid_acts = nn.functional.one_hot(exp_labels)
            M = torch.cdist(iid_acts.float(), ood_acts, p=1)
            weights = torch.as_tensor([])
            est = ( ot.emd2(weights, weights, M, numItermax=1e8, numThreads=8) / 2 ).item()
    
    elif metric in ['COTT-MC', 'COTT-NE']:
        threshold = get_threshold(model, val_iid_loader, n_class, args)
        batch_size = min(10000, n_test_sample)
        n_batch = math.ceil( n_test_sample // batch_size )
        
        print(
            f'total of {n_test_sample} test samples, running {n_batch} batches.'
        )
        
        if n_batch > 1:
            est = 0
            random.seed(0)
            cost_dist = []
            for _ in range(n_batch):
                rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                ood_acts_batch = ood_acts[rand_inds]
                
                exp_labels_batch = sample_label_dist(dsname, n_class, batch_size)
                iid_acts_batch = nn.functional.one_hot(exp_labels_batch)
                
                M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                
                weights = torch.as_tensor([])
                Pi = ot.emd(weights, weights, M, numItermax=1e8)
                
                if metric == 'COTT-MC':
                    costs = ( Pi * M.shape[0] * M ).sum(1) * -1
                
                elif metric == 'COTT-NE':
                    matched_ood_acts_batch = ood_acts_batch[torch.argmax(Pi, dim=1)]
                    matched_acts = (matched_ood_acts_batch + iid_acts_batch) / 2
                    costs = ( matched_acts * torch.log2( matched_acts ) ).sum(1)
                
                est = est + (costs < threshold).sum().item() / batch_size
                cost_dist.append(costs)
            
            est = est / n_batch
            cost_dist = torch.sort(torch.cat(cost_dist, dim=0))[0].tolist()
        
        else:
            torch.manual_seed(0)
            exp_labels = sample_label_dist(dsname, n_class, n_test_sample)
            iid_acts = nn.functional.one_hot(exp_labels)
            
            M = torch.cdist(iid_acts.float(), ood_acts, p=1)
            
            weights = torch.as_tensor([])
            Pi = ot.emd(weights, weights, M, numItermax=1e8)
            
            if metric == 'COTT-MC':
                costs = ( Pi * M.shape[0] * M ).sum(1) * -1
            elif metric == 'COTT-NE':
                matched_ood_acts = ood_acts[torch.argmax(Pi, dim=1)]
                matched_acts = (matched_ood_acts + iid_acts) / 2
                costs = ( matched_acts * torch.log2( matched_acts ) ).sum(1)
            
            est = (costs < threshold).sum().item() / batch_size
            cost_dist = torch.sort(costs)[0].tolist()
    
    elif metric == 'DCOT':
        if args.pretrained:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/pretrained_dcot_base.json"
        else:
            cache_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/scratch_dcot_base.json"
        
        if not os.path.exists(cache_dir):
            torch.manual_seed(0)
            exp_labels = sample_label_dist(dsname, n_class, len(ood_acts))
            label_acts = nn.functional.one_hot(exp_labels)
            
            M = torch.max( torch.abs( iid_acts.unsqueeze(1) - label_acts.unsqueeze(0) ), dim=-1)[0]
            weights = torch.as_tensor([])
            base_est = ( ot.emd2( weights, weights, M, numItermax=1e8, numThreads=8 ) ).item()
        
            with open(cache_dir, 'w') as f:
                json.dump({'base': base_est}, f)
        else:
            with open(cache_dir, 'r') as f:
                base_est = json.load(f)['base']
        
        M2 = torch.max( torch.abs( ood_acts.unsqueeze(1) - iid_acts.unsqueeze(0) ), dim=-1)[0]
        weights = torch.as_tensor([])
        add_est = ( ot.emd2( weights, weights, M2, numItermax=1e8, numThreads=8 ) ).item()
        
        print('base est:', base_est)
        print('add est:', add_est)
        print('iid error:', 1 - iid_acc)
        
        est = base_est + add_est
    
    elif metric == 'SCOTT':
        t = get_threshold(model, val_iid_loader, n_class, args)
        torch.manual_seed(10)
        slices = torch.randn(8, n_class)
        slices = torch.stack([slice / torch.sqrt( torch.sum( slice ** 2 ) ) for slice in slices], dim=0)
        
        exp_labels = sample_label_dist(dsname, n_class, len(ood_acts))
        iid_acts = nn.functional.one_hot(exp_labels)
        
        iid_act_scores = iid_acts.float() @ slices.T
        ood_act_scores = ood_acts.float() @ slices.T
        scores = torch.abs( torch.sort(ood_act_scores, dim=0)[0] - torch.sort(iid_act_scores, dim=0)[0] )
        est = ( scores > t ).sum().item() / len(ood_acts) / len(slices)
    
    print('------------------')
    print('True OOD error:', 1 - ood_acc)
    print(f'{metric} predicted OOD error:', est)
    print(f'MAE: {abs(1 - ood_acc - est)}')
    print(f'Time: {time.time() - start}')
    print('------------------')
    print()

    n_test_str = args.n_test_samples
    if pretrained:
        result_dir = f"results/{dsname}/pretrained/{args.arch}_{model_seed}-{model_epoch}/{metric}_{n_test_str}/{corruption}.json"
    else:
        result_dir = f"results/{dsname}/scratch/{args.arch}_{model_seed}-{model_epoch}/{metric}_{n_test_str}/{corruption}.json"

    print(result_dir)
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
    
    if metric in ['ATC-MC', 'ATC-NE', 'COTT-MC', 'COTT-NE']:
        if args.pretrained:
            cost_dist_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/{metric}_costs/pretrained_{corruption}.json"
        else:
            cost_dist_dir = f"cache/{dsname}/{args.arch}_{model_seed}-{model_epoch}/{metric}_costs/scratch_{corruption}.json"
        
        os.makedirs(os.path.dirname(cost_dist_dir), exist_ok=True)
        
        if not os.path.exists(cost_dist_dir):
            with open(cost_dist_dir, 'w') as f:
                json.dump([], f)

        with open(cost_dist_dir, 'r') as f:
            saved_costs = json.load(f)

        saved_costs.append({
            'costs': cost_dist, 
            't': threshold,
            'ood error': 1 - ood_acc,
            'iid error': 1 - iid_acc,
            'pred': ood_preds.tolist(),
            'pseudo-source shift': sum(abs(np.array(ood_preds_dist) - np.array(iid_tars_dist))) / 2,
            'pseudo-target shift': sum(abs(np.array(ood_preds_dist) - np.array(ood_tars_dist))) / 2
        })
        
        with open(cost_dist_dir, 'w') as f:
            json.dump(saved_costs, f)


if __name__ == "__main__":
    main()


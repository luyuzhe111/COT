import torch
import torch.nn as nn
import os
import json
import pickle
from tqdm import tqdm
import ot
import time
import numpy as np
from torch_datasets.configs import get_expected_label_distribution

def evaluation(net, testloader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total


def baseline_evaluation(net, testloader, val_loader, t, t_vec, net2):
    net.eval()
    net2.eval()
    criterion = nn.CrossEntropyLoss()
    metrics = torch.tensor([0.0] * 4)
    test_loss = 0
    correct = 0
    total = 0

    print('compute baselines...')
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            outputs2 = net2(inputs)

            # ConfScore
            softmax = nn.Softmax(dim=1)
            loss = torch.max(softmax(outputs), dim=1)[0]
            metrics[0] += loss.sum().item()

            # Entropy
            loss = torch.sum(-softmax(outputs) * torch.log2(softmax(outputs)), dim=1)
            metrics[1] += loss.sum().item()

            # ATC
            s_softmax = torch.sum(softmax(outputs) * torch.log2(softmax(outputs)), dim=1)
            loss = s_softmax < t
            metrics[2] += loss.sum().item()

            # Test loss
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            _, predicted2 = outputs2.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            # AgreeScore
            metrics[3] += predicted.eq(predicted2).sum().item()

    return metrics / total, test_loss / total, 100 * correct / total
    

def compute_t(net, iid_loader):
    net.eval()
    misclassified = 0
    res = []
    with torch.no_grad():
        for _, items in enumerate(tqdm(iid_loader)):
            inputs, targets = items[0], items[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            misclassified += targets.size(0) - predicted.eq(targets).sum().item()
            softmax = nn.Softmax(dim=1)
            res.append(torch.sum(softmax(outputs) * torch.log2(softmax(outputs)), dim=1))
    s_softmax = torch.cat(res)

    sorted_s_softmax = torch.sort(s_softmax)[0]
    return sorted_s_softmax[misclassified - 1] + 1e-9


def compute_st(net, iid_loader, n_class):
    net.eval()
    softmax_vecs = []
    preds, tars = [], []
    with torch.no_grad():
        for _, items in enumerate(tqdm(iid_loader)):
            inputs, targets = items[0], items[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, prediction = outputs.max(1)

            preds.extend( prediction.tolist() )
            tars.extend( targets.tolist() )
            softmax_vecs.append( nn.functional.softmax(outputs, dim=1).cpu() )
    
    preds, tars  = torch.as_tensor(preds), torch.as_tensor(tars)
    softmax_vecs = torch.cat(softmax_vecs, dim=0)
    target_vecs = nn.functional.one_hot(tars)
    
    torch.manual_seed(10)
    slices = torch.randn(8, n_class)
    slices = torch.stack([slice / torch.sqrt( torch.sum( slice ** 2 ) ) for slice in slices], dim=0)
    
    iid_act_scores = softmax_vecs.float() @ slices.T
    ood_act_scores = target_vecs.float() @ slices.T
    scores = torch.sort( torch.abs( torch.sort(ood_act_scores, dim=0)[0] - torch.sort(iid_act_scores, dim=0)[0] ), dim=0 )[0]
    n_correct = preds.eq(tars).sum()
    t = scores[n_correct - 1]
    return t
    

def get_threshold(net, iid_loader, n_class, args):
    dsname = args.dataset
    arch = args.arch
    model_seed = args.model_seed
    model_epoch = args.ckpt_epoch
    metric = args.metric
    pretrained = args.pretrained
    cost = args.cost
    
    if metric == 'ATC':
        if pretrained:
            cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/pretrained_atc_threshold.json"
        else:
            cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/scratch_atc_threshold.json"
        
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                t = data['t']
        else:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            with open(cache_dir, 'w') as f:
                print('compute confidence threshold...')
                t = compute_t(net, iid_loader).item()
                json.dump({'t': t}, f)
        
        return t
    
    elif metric == 'COTT':
        if pretrained:
            cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/pretrained_cott-{cost}_threshold.json"
        else:
            cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/scratch_cott-{cost}_threshold.json"
        
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                t = data['t']
        else:
            with open(cache_dir, 'w') as f:
                print('compute confidence threshold...')
                t = compute_cott(net, iid_loader, n_class, cost)
                json.dump({'t': t}, f)
        
        return t

    elif metric == 'SCOTT':
        t = compute_st(net, iid_loader, n_class)
        
        return t
    else:
        raise ValueError(f'unknown metric {metric}')
        

def compute_cott(net, iid_loader, n_class, cost):
    net.eval()
    softmax_vecs = []
    preds, tars = [], []
    with torch.no_grad():
        for _, items in enumerate(tqdm(iid_loader)):
            inputs, targets = items[0], items[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, prediction = outputs.max(1)

            preds.extend( prediction.tolist() )
            tars.extend( targets.tolist() )
            softmax_vecs.append( nn.functional.softmax(outputs, dim=1).cpu() )
    
    preds, tars  = torch.as_tensor(preds), torch.as_tensor(tars)
    softmax_vecs = torch.cat(softmax_vecs, dim=0)
    target_vecs = nn.functional.one_hot(tars)
    
    max_n = 10000
    if len(target_vecs) > max_n:
        print(f'sampling {max_n} out of {len(target_vecs)} validation samples...')
        torch.manual_seed(0)
        rand_inds = torch.randperm(len(target_vecs))
        tars = tars[rand_inds][:max_n]
        preds = preds[rand_inds][:max_n]
        target_vecs = target_vecs[rand_inds][:max_n]
        softmax_vecs = softmax_vecs[rand_inds][:max_n]

    print('computing assignment...')
    if cost == 'L1':
        M = torch.cdist(target_vecs.float(), softmax_vecs, p=1)
    elif cost == 'L2':
        M = torch.cdist(target_vecs.float(), softmax_vecs, p=2)
    
    start = time.time()
    weights = torch.as_tensor([])
    Pi = ot.emd(weights, weights, M, numItermax=1e8)

    print(f'done. {time.time() - start}s passed')
    
    costs = ( Pi * M.shape[0] * M ).sum(1)
    
    n_correct = preds.eq(tars).sum()
    
    t = torch.sort( costs )[0][n_correct - 1].item()
    
    return t


def gather_outputs(model, dataloader, device, cache_dir):
    if os.path.exists(cache_dir):
        print('loading cached result from', cache_dir)
        data = pickle.load( open(cache_dir, "rb" ))
        acts, preds, tars = data['act'], data['pred'], data['tar']
        return acts.to(device), preds.to(device), tars.to(device)
    else:
        preds = []
        acts = []
        tars = []
        print('computing result for', cache_dir)
        with torch.no_grad():
            for items in tqdm(dataloader):
                inputs, targets = items[0], items[1]               
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)

                act = outputs

                preds.append(predicted)
                acts.append(act)
                tars.extend(targets)
        
        act, pred, tar = torch.concat(acts), torch.concat(preds), torch.as_tensor(tars, device=device)

        data = {'act': act.cpu(), 'pred': pred.cpu(), 'tar': tar.cpu()}
        pickle.dump( data, open( cache_dir, "wb" ) )

    return act, pred, tar


class HistogramDensity: 
    def _histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
    
    def __init__(self, num_bins = 10, equal_mass=False):
        self.num_bins = num_bins 
        self.equal_mass = equal_mass
        
        
    def fit(self, vals): 
        
        if self.equal_mass:
            self.bins = self._histedges_equalN(vals, self.num_bins)
        else: 
            self.bins = np.linspace(0,1.0,self.num_bins+1)
    
        self.bins[0] = 0.0 
        self.bins[self.num_bins] = 1.0
        
        self.hist, bin_edges = np.histogram(vals, bins=self.bins, density=True)
    
    def density(self, x): 
        curr_bins = np.digitize(x, self.bins, right=True)
        
        curr_bins -= 1
        return self.hist[curr_bins]


def get_im_estimate(probs_source, probs_target, corr_source): 
    probs_source = probs_source.numpy()
    probs_target = probs_target.numpy()
    corr_source = corr_source.numpy()

    source_binning = HistogramDensity()
    
    source_binning.fit(probs_source)
    
    target_binning = HistogramDensity()
    target_binning.fit(probs_target)
    
    weights = target_binning.density(probs_source) / source_binning.density(probs_source)
    weights = weights/ np.mean(weights)
    
    return 1 - np.mean(weights * corr_source)
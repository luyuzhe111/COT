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


# ----------- helper functions to find threshold -----------
    
def get_threshold(net, iid_loader, n_class, args):
    dsname = args.dataset
    arch = args.arch
    model_seed = args.model_seed
    model_epoch = args.ckpt_epoch
    metric = args.metric
    pretrained = args.pretrained
    if pretrained:
        cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/pretrained_ts_{metric}_threshold.json"
    else:
        cache_dir = f"cache/{dsname}/{arch}_{model_seed}-{model_epoch}/scratch_ts_{metric}_threshold.json"
    
    if metric in ['ATC-MC', 'ATC-NE']:
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                t = data['t']
        else:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            print('compute confidence threshold...')
            t = compute_t(net, iid_loader, metric).item()
            with open(cache_dir, 'w') as f:
                json.dump({'t': t}, f)
        
        return t
    
    elif metric in ['COTT-MC', 'COTT-NE', 'COTT-val-MC']:
        if os.path.exists(cache_dir):
            with open(cache_dir, 'r') as f:
                data = json.load(f)
                t = data['t']
        else:
            print('compute confidence threshold...')
            t = compute_cott(net, iid_loader, n_class, metric)
            with open(cache_dir, 'w') as f:
                json.dump({'t': t}, f)
        
        return t

    elif metric == 'SCOTT':
        t = compute_sliced_t(net, iid_loader, n_class)
        return t
    else:
        raise ValueError(f'unknown metric {metric}')
        

def compute_cott(net, iid_loader, n_class, metric):
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
    
    if metric == 'COTT-val-MC':
        n_incorrect = preds.ne(tars).sum()
        costs = (1 - softmax_vecs.amax(1)) * 2 * -1
        t = torch.sort( costs )[0][n_incorrect - 1].item()
    else:
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
        M = torch.cdist(target_vecs.float(), softmax_vecs, p=1)
        
        start = time.time()
        weights = torch.as_tensor([])
        Pi = ot.emd(weights, weights, M, numItermax=1e8)

        print(f'done. {time.time() - start}s passed')
        if metric == 'COTT-MC':
            costs = ( Pi * M.shape[0] * M ).sum(1) * -1
        elif metric == 'COTT-NE':
            matched_softmax = softmax_vecs[torch.argmax(Pi, dim=1)]
            matched_acts = (matched_softmax + target_vecs) / 2
            costs = ( matched_acts * torch.log2(matched_acts) ).sum(1)
        
        n_incorrect = preds.ne(tars).sum()
        t = torch.sort( costs )[0][n_incorrect - 1].item()
        
    return t


def compute_t(net, iid_loader, metric):
    net.eval()
    misclassified = 0
    mc = []
    ne = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for _, items in enumerate(tqdm(iid_loader)):
            inputs, targets = items[0], items[1]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            misclassified += targets.size(0) - predicted.eq(targets).sum().item()
            
            ne.append(torch.sum(softmax(outputs) * torch.log2(softmax(outputs)), dim=1))
            mc.append(softmax(outputs).max(1)[0])
    
    ne = torch.cat(ne)
    mc = torch.cat(mc)
    
    if metric == 'ATC-MC':
        t = torch.sort(mc)[0][misclassified - 1]
    elif metric == 'ATC-NE':
        t = torch.sort(ne)[0][misclassified - 1]
    return t


def compute_sliced_t(net, iid_loader, n_class):
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


# ----------- helper functions for evaluation -----------

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


def get_temp_dir(cache_dir, seed, model_epoch, opt_bias=False):
    if opt_bias:
        temp_dir = f"{cache_dir}/base_model_{seed}-{model_epoch}_temp_with_bias.json"
    else:
        temp_dir = f"{cache_dir}/base_model_{seed}-{model_epoch}_temp.json"
    
    return temp_dir


# ----------- helper code for other baselines -----------

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
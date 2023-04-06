import torch
import torch.nn as nn
import os
import pickle
from tqdm import tqdm
import ot

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





def compute_cott(net, iid_loader, n_class):
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
    
    thresholds = torch.zeros(n_class)
    preds, tars  = torch.as_tensor(preds), torch.as_tensor(tars)
    softmax_vecs = torch.cat(softmax_vecs, dim=0)
    target_vecs = nn.functional.one_hot(tars)

    print('computing assignment...')
    M = torch.sum(
        torch.abs( target_vecs.unsqueeze(1) - softmax_vecs.unsqueeze(0) ), dim=-1 
    )
    weights = torch.as_tensor([])
    Pi = ot.emd(weights, weights, M, numItermax=10**8)
    label_inds = Pi.nonzero()[:, 0]
    matched_softmax_inds = Pi.nonzero()[:, 1]
    print('done.')

    for i in range(n_class):
        clss_tar_inds = ( tars == i )
        n_correct = (preds[clss_tar_inds]).eq(tars[clss_tar_inds]).sum()
        clss_scores = torch.sort (
            M[ label_inds[clss_tar_inds], matched_softmax_inds[clss_tar_inds] ]
        )[0]
        thresholds[i] = clss_scores[n_correct - 1]

    return thresholds


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
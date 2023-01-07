import torch
import torch.nn as nn
import os
import pickle
from tqdm import tqdm
import contextlib


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


def baseline_evaluation(net, testloader, val_loader, t, t_vec):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    metrics = torch.tensor([0.0] * 3)
    test_loss = 0
    correct = 0
    total = 0
    print('compute baselines...')
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

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
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return metrics / total, test_loss / total, 100. * correct / total

def compute_t(net, iid_loader):
    net.eval()
    misclassified = 0
    res = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(iid_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            misclassified += targets.size(0) - predicted.eq(targets).sum().item()
            softmax = nn.Softmax(dim=1)
            res.append(torch.sum(softmax(outputs) * torch.log2(softmax(outputs)), dim=1))
    s_softmax = torch.cat(res)

    sorted_s_softmax = torch.sort(s_softmax)[0]
    return sorted_s_softmax[misclassified - 1] + 1e-9


def compute_t_vec(net, iid_loader):
    net.eval()
    misclassified = 0
    res = []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(iid_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            misclassified += targets.size(0) - predicted.eq(targets).sum().item()
            res.append(outputs)
    
    logits = torch.cat(res)

    sorted_logits = torch.sort(logits, dim=0)[0]
    return sorted_logits[misclassified - 1, :] + 1e-9


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
            for inputs, targets in tqdm(dataloader):
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


import torch
import contextlib

class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


interp1d = Interp1d.apply

def interpolate(p, q):
    # p, q are of shape N x d
    if len(q) == len(p):
        return p, q
    
    if len(p) > len(q):
        ref, x = q, p

    else:
        ref, x = p, q
    
    device = x.device
    
    x_domain = torch.linspace(0, 1, x.shape[0]).repeat(x.shape[1], 1).to(device)
    ref_domain = torch.linspace(0, 1, ref.shape[0]).repeat(ref.shape[1], 1).to(device)
    x_interp_range = interp1d(x_domain, x.T, ref_domain).T

    return ref, x_interp_range
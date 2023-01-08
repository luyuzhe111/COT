import argparse
import json
from load_data import *
from utils import baseline_evaluation, compute_t

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
parser.add_argument('--corruption_path', default='./data/CIFAR-10-C', type=str)
parser.add_argument('--data_type', default='cifar-10', type=str)
parser.add_argument('--corruption', default='snow', type=str)
parser.add_argument('--severity', default=5, type=int)
parser.add_argument('--pseudo_iters', default=50, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_base_model', action='store_true',
                    default=False, help='apply base_model for computing ProjNorm')
args = vars(parser.parse_args())
print(args)

if __name__ == "__main__":
    # setup valset_iid/val_ood loaders
    random_seeds = torch.randint(0, 10000, (2,))
    # random_seeds = torch.tensor([0, 1])

    data_type = args['data_type']

    valset_iid = load_image_dataset(corruption_type='clean',
                                    clean_path=args['data_path'],
                                    corruption_path=args['corruption_path'],
                                    corruption_severity=0,
                                    datatype='test',
                                    type=data_type,
                                    seed=random_seeds[0])
    
    val_iid_loader = torch.utils.data.DataLoader(valset_iid,
                                                 batch_size=args['batch_size'],
                                                 shuffle=False)

    valset_ood = load_image_dataset(corruption_type=args['corruption'],
                                    clean_path=args['data_path'],
                                    corruption_path=args['corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype='test',
                                    type=data_type,
                                    seed=random_seeds[1])
    
    val_ood_loader = torch.utils.data.DataLoader(valset_ood,
                                                 batch_size=args['batch_size'],
                                                 shuffle=False)

    # init ProjNorm
    save_dir_path = f"./checkpoints/{data_type}/{args['arch']}"
    base_model = torch.load('{}/base_model.pt'.format(save_dir_path))
    base_model.eval()


    corruption = args['corruption']
    severity = args['severity']
    result_dir = f"results/{os.path.basename(args['data_path'])}/{args['arch']}"
    
    cache_dir = f"cache/{os.path.basename(args['data_path'])}/{args['arch']}/iid_result.json"
    if os.path.exists(cache_dir):
        with open(cache_dir, 'r') as f:
            data = json.load(f)
            t = data['t']
    else:
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)

        with open(cache_dir, 'w') as f:
            print('compute confidence threshold...')
            t = compute_t(base_model, val_iid_loader).item()
            json.dump({'t': t}, f) 
        
    print(f"===========model={args['arch']}, type={args['corruption']}, severity={args['severity']}===========")
    metrics, test_loss_ood, test_acc_ood = \
        baseline_evaluation(net=base_model, testloader=val_ood_loader, iid_loader=val_iid_loader, t=t)
    
    metrics = metrics.tolist()
    
    print('Test Loss: ', test_loss_ood)
    print('(out-of-distribution) test acc: ', test_acc_ood)
    print('========Metrics========')
    print('Rotation: ', metrics[0])
    print('ConfScore: ', metrics[1])
    print('Entropy: ', metrics[2])
    print('ATC: ', metrics[3])
    print('========Finished========')

    def save_json(result_dir, method, corruption, severity, ood_acc, ood_metric):
        save_dir = f"{result_dir}/{method}/{corruption}.json"
        print(result_dir, os.path.dirname(save_dir), os.path.basename(save_dir))
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        
        if not os.path.exists(save_dir):
            with open(save_dir, 'w') as f:
                data = [{
                    'corruption': corruption,
                    'corruption level': severity,
                    'method': method,
                    'metric': ood_metric,
                    'acc': float(ood_acc),
                    'error': 1 - ood_acc
                }]
                json.dump(data, f)
        else:
            with open(save_dir, 'r') as f:
                data = json.load(f)
            
            data.append({
                'corruption': corruption,
                'corruption level': severity,
                'method': method,
                'metric': ood_metric,
                'acc': float(ood_acc),
                'error': 1 - ood_acc
            })

            with open(save_dir, 'w') as f:
                json.dump(data, f)
    
    save_json(result_dir, 'Rotation', corruption, severity, test_acc_ood / 100, metrics[0])
    save_json(result_dir, 'ConfScore', corruption, severity, test_acc_ood / 100, metrics[1] * -1)
    save_json(result_dir, 'Entropy', corruption, severity, test_acc_ood / 100, metrics[2])
    save_json(result_dir, 'ATC', corruption, severity, test_acc_ood / 100, metrics[3])




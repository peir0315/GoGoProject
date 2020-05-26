import argparse

import csv
from os.path import join as PJ
import torch
from torch.utils.data import DataLoader

from datasets.Dataset import CIFAR10Dataset
from networks import ResNet18, HaHaNet

from utils import config, data_transform
from datasets.evaluation import accuracy


def test(dataloader, model):
    model.eval()
    results = {'predict': [], 'label': []}
    for it, (labels, images) in enumerate(dataloader):

        # Drop images and labels into GPU
        images = images.detach()
        labels = labels.detach()

        # Take class with largest score as predict
        predicts = model(images)
        predicts = torch.argmax(predicts, 1)

        # Record results
        results['predict'] += predicts.tolist()
        results['label'] += labels.tolist()

        if (it + 1) % (len(dataloader) / 10) == 0:
            print(f"it: [{it+1:03d}/{len(dataloader):03d}]", end='\r')
    acc = accuracy(results['label'], results['predict'])
    return acc


if __name__ == '__main__':
    ########################################
    # Environment and Experiment setting
    ########################################
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, default='exp1_ResNet18')
    parser.add_argument('--model_weights', '-w', type=str, default='')
    args = parser.parse_args()

    # Load experiment config
    config_path = f"./configs/{args.exp}.yaml"
    config = config(config_path)
    exp_name = config['exp_name']
    print(f"EXP: {exp_name}")

    save_root = PJ(f"./results", exp_name)

    # Show some experiment info
    model_weights = args.model_weights
    print(f"Model: {config['model']}, Weights: {model_weights}")
    print(f"Pretrained: {config['pretrained']}, Freeze: {config['freeze']}")

    ########################################
    # Data loader
    ########################################
    class2idx = config['class2idx']
    idx2class = {class2idx[k]: k for k in class2idx.keys()}
    # Dataset
    transform = data_transform(config, train=False)
    testset = CIFAR10Dataset(config['data_root'], config['test_file'], class2idx, transform)
    # Dataloader
    testloader = DataLoader(testset, config['test_size'], shuffle=False, num_workers=config['num_workers'])

    ########################################
    # Model
    ########################################
    num_class = len(class2idx)
    input_size = config['new_size']
    if config['model'] == 'ResNet18':
        model = ResNet18(input_size, num_class, config['pretrained'], config['freeze'])
    elif config['model'] == 'HaHaNet':
        model = HaHaNet(input_size, num_class, config['pretrained'], config['freeze'])
    else:
        raise f"Model {config['model']} is not support."

    # Drop model into GPU
    model.cuda()

    ########################################
    # Loading model
    ########################################
    model.load_state_dict(torch.load(PJ(save_root, model_weights)))
    epoch = int(model_weights[:4])
    print(f"Loading model {model_weights} successed!")

    ########################################
    # Start evaluate (testing set)
    ########################################
    model.eval()
    torch.set_grad_enabled(False)
    acc = test(testloader, model)
    print(f"Accuracy: {acc:0.4f}\n")

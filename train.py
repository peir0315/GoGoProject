import argparse
import os
from os.path import join as PJ
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tensorboardX import SummaryWriter

from datasets.Dataset import CIFAR10Dataset
from networks import ResNet18, HaHaNet
from utils import config, data_transform
from test import test


########################################
# Environment and Experiment setting
########################################
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', '-e', type=str, default='exp1_ResNet18')
args = parser.parse_args()

# Load experiment config
config_path = PJ(os.getcwd(), "configs", f"{args.exp}.yaml")
config = config(config_path)
exp_name = config['exp_name']
print(f"EXP: {exp_name}")

# Create saving directory
save_root = PJ(os.getcwd(), "results", exp_name)
if not os.path.exists(save_root):
    os.makedirs(save_root)
    print(f"Create {save_root}")
# Tensorboad
writer = SummaryWriter(PJ(os.getcwd(), "results", "logs", exp_name))
# Saving config file
shutil.copy(config_path, PJ(save_root, f"{exp_name}.yaml"))

# Show some experiment info
print(f"Model: {config['model']}")
print(f"Pretrained: {config['pretrained']}, Freeze: {config['freeze']}")

########################################
# Data loader
########################################
class2idx = config['class2idx']
idx2class = {class2idx[k]: k for k in class2idx.keys()}
# Dataset
transform = data_transform(config, train=True)
trainset = CIFAR10Dataset(config['data_root'], config['train_file'], class2idx, transform)
transform = data_transform(config, train=False)
valset = CIFAR10Dataset(config['data_root'], config['val_file'], class2idx, transform)

# Dataloader
trainloader = DataLoader(trainset, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
valloader = DataLoader(valset, config['test_size'], shuffle=False, num_workers=config['num_workers'])

print(f"Train data: {len(trainset)} | Val data: {len(valset)}")
print(f"Number of batch: {len(trainloader)}/epoch")

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
#model.cuda()
model
########################################
# Criterion and Optimizer
########################################
def criterion(predicts, targets):
    """ predicts: (N, C), targets: (N) """
    ce = nn.CrossEntropyLoss()
    loss = ce(predicts, targets.reshape(-1))
    return loss

optimizer = Adam(model.parameters(), config['lr'],
                 weight_decay=config['weight_decay'])

resume = config['resume']
if resume:
    model_weights = os.listdir(save_root)
    model_weights.remove("optimizer.pt")
    model_weights.sort()
    last_model_name = model_weights[-1]

    model.load_state_dict(torch.load(PJ(save_root, last_model_name)))
    optimizer.load_state_dict(torch.load(PJ(save_root, "optimizer.pt")))
    print(f"Loading model {last_model_name} successed!")
    print(f"Loading optimizer successed!")
    iteration, last_epoch = int(last_model_name[-11:-3]), int(last_model_name[:4])
iteration, last_epoch = (iteration, last_epoch) if resume else (0, 1)

# Learning rate decay scheduler
scheduler = lr_scheduler.MultiStepLR(
    optimizer, config["step_size"], config["gamma"], last_epoch=last_epoch - 2)

########################################
# Start training
########################################
print("\n> Training")
for epoch in range(last_epoch, config["max_epoch"]+1):
    # scheduler step in each epoch
    scheduler.step()
    if (__name__ == '__main__'):
        #freeze_support()
        for it, (labels, images) in enumerate(trainloader):
            optimizer.zero_grad()

            # Drop images and labels into GPU
            images = images.detach()
            labels = labels.detach()

            # Forward
            predicts = model(images)

            # Backward
            loss = criterion(predicts, labels)
            # 1. Calculate gradient
            loss.backward()
            # 2. Update parameters
            optimizer.step()

            # Record training log (Tensorboard)
            if (iteration + 1) % config['log_iter'] == 0:
                print(f"Iteration: [{it + 1:06d} / {len(trainloader):06d}]  (Epoch: {epoch})")
                writer.add_scalar("loss", loss.item(), iteration + 1)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], iteration + 1)
            iteration += 1

        # Saving model
            torch.save(model.state_dict(), PJ(save_root, f"{epoch:04d}_{iteration:08d}.pt"))
        torch.save(optimizer.state_dict(), PJ(save_root, "optimizer.pt"))
        print(f"Saving model in {save_root} finished.\n")

        ########################################
        # Start evaluate (validation set)
        ########################################
        # Evaluate mode. Take of dropout.
        model.eval()
        # Not calculate gradient
        torch.set_grad_enabled(False)
        print("> Evaluation")

        accuracy = test(valloader, model)
        print(f"[{epoch}] Accuracy: {accuracy:0.4f}\n")
        writer.add_scalar("Evaluate/accuracy_val", accuracy, epoch)

        # Training mode.
        model.train()
        torch.set_grad_enabled(True)

import yaml
from torchvision import transforms


def data_transform(config, train=True):
    transform_list = [transforms.Resize(config['new_size']),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    return transform


def config(config_path):
    """ Loading config file. """
    with open(config_path, 'r') as f:
        return yaml.load(f)

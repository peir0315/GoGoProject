from torch import nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, input_size, num_class, pretrained, freeze=False, init=None):
        super().__init__()
        model = resnet18(pretrained)
        model = self.freeze_backbone(model) if freeze else model

        # Change avgpool layer (different input size)
        pool_dim = (input_size[0] // 2**5, input_size[1] // 2**5)
        model.avgpool = nn.AvgPool2d(pool_dim, stride=1, padding=0)
        # Change last Fc-layer (different num class)
        model.fc = nn.Linear(512, num_class)
        self.model = model

    def forward(self, x):
        predict = self.model(x)
        return predict

    def freeze_backbone(self, model):
        """ Don't calculate gradients when requires_grad is False. """
        for param in model.parameters():
            param.requires_grad = False
        return model


class HaHaNet(nn.Module):
    def __init__(self, input_size, num_class, init=None):
        super().__init__()
        self.convs = nn.Sequential(
            # nn.Conv2d(in_channel, out_channel, kernel_size, strid, padding)
            nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(8), nn.Dropout2d(0.5),
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(32),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(8), nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(32))

        self.fc = nn.Sequential(
            nn.Linear(32, 10), nn.ReLU(), nn.BatchNorm2d(8), nn.Dropout2d(0.5),
            nn.Linear(32, 10))

    def forward(self, x):
        features = self.convs(x).reshape(x.shape[0], -1)
        predicts = self.fc(features)
        return predicts

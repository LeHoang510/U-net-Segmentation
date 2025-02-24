import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from utils.utils import *
from utils.display_pred import display_prediction
from model.unet import UNet


def test():
    set_seed(555)
    img_size = (128, 128)
    num_classes = 3

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    test_set = OxfordIIITPet(root="pets_data", split="test", target_types="segmentation",
                                download=True, transform=transform, target_transform=target_transform)

    model = UNet(n_channels=3, n_classes=num_classes)
    model.load_state_dict(torch.load(osp.join("output", "unet_model.pth")))

    for i in range(10):
        img, gt = test_set[i]
        display_prediction(model, img, gt, num_classes, img_size)


if __name__ == "__main__":
    test()
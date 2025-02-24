import copy

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

from utils.utils import *
from utils.tensorboard import Logger
from model.unet import UNet



def train():
    set_seed(555)
    img_size = (128, 128)
    num_classes = 3

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])


    train_set = OxfordIIITPet(root="pets_data", split="trainval", target_types="segmentation", 
                            download=True, transform=transform, target_transform=target_transform)

    test_set = OxfordIIITPet(root="pets_data", split="test", target_types="segmentation",
                                download=True, transform=transform, target_transform=target_transform)

    batch_size = 64
    num_workers = 28
    train_loader = DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = UNet(num_classes=num_classes)
    model.to(device)

    logger = Logger(model, None, log_dir=osp.join("output", "unet_logs"))

    max_epoch = 30 
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_index = 80
    display_image = test_set[test_index][0]
    display_target = test_set[test_index][1]

    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0 
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        logger.write_dict(epoch+1, max_epoch, train_loss, test_loss=test_loss)

    return train_losses, test_losses

if __name__ == "__main__":
    train()
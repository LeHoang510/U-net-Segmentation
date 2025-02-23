from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torch


img_size = (128, 128)
num_classes = 3

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

def target_transform(target):
    img = transforms.Resize(img_size)(target)
    img = transforms.functional.pil_to_tensor(img).squeeze()
    img = img-1
    img = img.to(torch.long)
    return img

train_set = OxfordIIITPet(root="pets_data", split="trainval", target_types="segmentation", 
                          download=True, transform=transform, target_transform=target_transform)

test_set = OxfordIIITPet(root="pets_data", split="test", target_types="segmentation",
                            download=True, transform=transform, target_transform=target_transform)

batch_size = 64
num_workers = 28
train_loader = DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)



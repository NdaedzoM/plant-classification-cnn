from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, device='cpu'):
        self.root = Path(root_dir)
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.class_aliases = {
            'Black-grass': 'BG', 'Charlock': 'CH', 'Cleavers': 'CL',
            'Common Chickweed': 'CW', 'Common wheat': 'WH', 'Fat Hen': 'FH',
            'Loose Silky-bent': 'LS', 'Maize': 'MA', 'Scentless Mayweed': 'SC',
            'Shepherds Purse': 'SH', 'Small-flowered Cranesbill': 'SF', 'Sugar beet': 'SB'
        }
        self.original_to_alias = {
            'Black-grass': 'BG', 'Charlock': 'CH', 'Cleavers': 'CL',
            'Common Chickweed': 'CW', 'Common wheat': 'WH', 'Fat Hen': 'FH',
            'Loose Silky-bent': 'LS', 'Maize': 'MA', 'Scentless Mayweed': 'SC',
            'Shepherds Purse': 'SH', 'Small-flowered Cranesbill': 'SF', 'Sugar beet': 'SB'
        }
        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.glob('*.[jp][pn]g'):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image.to(self.device), torch.tensor(label, dtype=torch.long, device=self.device)

class TestDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, device='cuda'):
        self.root = Path(root_dir)
        self.transform = transform
        self.device = device
        self.images = sorted(list(self.root.glob('*.[jp][pn]g')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.to(self.device), img_path.name

def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_data_loaders(batch_size=32, val_split=0.2, device='cpu'):
    train_dir = Path('/content/drive/My Drive/cnn/data/train')
    test_dir = Path('/content/drive/My Drive/cnn/data/test')
    transform = get_transform()
    train_all_dataset = TrainDataset(train_dir, transform=transform, device=device)
    train_size = int(len(train_all_dataset) * (1 - val_split))
    val_size = len(train_all_dataset) - train_size
    train_data, val_data = random_split(train_all_dataset, [train_size, val_size])
    test_dataset = TestDataset(test_dir, transform=transform, device=device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_all_dataset.class_names, train_all_dataset.class_aliases, train_all_dataset.original_to_alias
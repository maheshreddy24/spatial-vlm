
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from scipy.io import loadmat
from icecream import ic


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split="trainval", transform=None):
        """
        root_dir: path to 'oxford-iiit-pet'
        split: 'trainval' or 'test'
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_file = os.path.join(root_dir, "annotations", f"{split}.txt")
        # self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], #[0.485, 0.456, 0.406]
                                     std=[0.229, 0.224, 0.225]) #[0.229, 0.224, 0.225]
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_annotations()
        ic(len(self.samples))

    def _load_annotations(self):
        with open(self.annotations_file, "r") as f:
            for line in tqdm(f):
                parts = line.strip().split()
                image_name = parts[0] + ".jpg"
                label = int(parts[1]) - 1  # convert to 0-based index
                
                self.samples.append((image_name, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label
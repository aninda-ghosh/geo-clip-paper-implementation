from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from PIL import Image as im
import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GeoCLIPDataset(Dataset):
    
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.dataset_metadata = dataset_path + "metadata.csv"
        self.transform = transform
        self.images, self.coordinates = self.load_dataset()

    def load_dataset(self):
        data_df = pd.read_csv(self.dataset_metadata)

        images = []
        coordinates = []

        for _, row in tqdm(data_df.iterrows(), desc="Loading image paths and coordinates"):
            images.append(self.dataset_path + "images/" +row['id'])
            coordinates.append((row['latitude'], row['longitude']))

        return images, coordinates
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        img = im.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        #convert everything to tensor
        gps = torch.tensor(gps, dtype=torch.float32)

        return img, gps


        
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import torch.nn.functional as F

from model.image_encoder import ImageEncoder
from model.location_encoder import LocationEncoder
from solver.losses import Contrastive_Loss


def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list

class GeoCLIP(pl.LightningModule):
    def __init__(self, cfg):
        super(GeoCLIP, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        
        self.criterion = Contrastive_Loss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.TRAINING.LEARNING_RATE, weight_decay=cfg.TRAINING.WEIGHT_DECAY)
        
        self.gps_gallery = torch.tensor(pd.read_csv(cfg.DATA.GPS_GALLERY)[['latitude', 'longitude']].values, dtype=torch.float32) 
        self.queue_size = cfg.MODEL.GPS_QUEUE_SIZE
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
        
    def forward(self, image, gps):
        image_features = self.image_encoder(image)
        gps_features = self.location_encoder(gps)
        logit_scale = self.logit_scale.exp()

        image_features = F.normalize(image_features, dim=1)
        gps_features = F.normalize(gps_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ gps_features.t())
        return logits_per_image

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        image, gps = batch

        gps_queue = self.get_gps_queue()

        gps_all = torch.cat([gps, gps_queue], dim=0)
        self.dequeue_and_enqueue(gps)

        output = self(image, gps_all)
        loss = self.criterion(output)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        image, gps = batch
        gps = torch.tensor(gps, dtype=torch.float32)
        gps_queue = self.get_gps_queue()

        gps_all = torch.cat([gps, gps_queue], dim=0)
        output = self(image, gps_all)
        loss = self.criterion(output)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
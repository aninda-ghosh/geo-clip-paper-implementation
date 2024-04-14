from config import cfg
from geo_clip import GeoCLIP, img_train_transform
from dataset.dataset import GeoCLIPDataset
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import StochasticWeightAveraging



if __name__ == "__main__":
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    torch.set_float32_matmul_precision('medium')

    # Initialize the dataset and dataloaders
    dataset = GeoCLIPDataset(dataset_path=cfg.DATA.TRAIN_DATASET_PATH, transform=img_train_transform())

    #truncate the dataset to 114,352 samples
    dataset = torch.utils.data.Subset(dataset, range(114352))
    
    train_size = int(cfg.TRAINING.TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.VALIDATION.NUM_WORKERS, persistent_workers=True)

    model = GeoCLIP(cfg)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        # detect_anomaly=True,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.TRAINING.SWA_LRS)]
        # gradient_clip_val=1.5
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
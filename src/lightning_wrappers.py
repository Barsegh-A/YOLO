import torch
import pytorch_lightning as pl

class YOLOV1LightningModule(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate=1e-3):        
        super(YOLOV1LightningModule, self).__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

    def forward(self, x):        
        return self.model(x)

    def training_step(self, batch, batch_idx):        
        images, targets = batch['image_tensor'], batch['yolo_targets']
        predictions = self.model(images)
        loss = self.criterion(predictions, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        images, targets = batch['image_tensor'], batch['yolo_targets']
        predictions = self.model(images)
        loss = self.criterion(predictions, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class VOCDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=64):        
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)



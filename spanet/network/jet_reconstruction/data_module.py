from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self, dataloader_options, train_dataset, val_dataset, test_dataset):
        super().__init__()
        self.dataloader_options = dataloader_options
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        print("train dataloader options", *self.dataloader_options.items())
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, drop_last=True, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_options)

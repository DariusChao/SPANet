from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.options import Options

class JetReconstructionDataModule(LightningDataModule):
    def __init__(self, options: Options, train_dataset, val_dataset, test_dataset):
        super().__init__()
        self.dataloader_options = {"batch_size": options.batch_size, "pin_memory": options.num_gpu > 0, "num_workers": options.num_dataloader_workers}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
    def prepare_data(self):
        pass
        
    def setup(self, stage: str):
        pass
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, drop_last=True, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_options)

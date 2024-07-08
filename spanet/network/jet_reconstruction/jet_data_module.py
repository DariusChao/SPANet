from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.options import Options

class DataModule(LightningDataModule):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.dataloader_options = {"batch_size": options.batch_size, "pin_memory": options.num_gpu > 0, "num_workers": options.num_dataloader_workers}
        # self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()
        
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        event_info_file = self.options.event_info_file
        training_file = self.options.training_file
        validation_file = self.options.validation_file

        training_range = self.options.dataset_limit
        validation_range = 1.0

        # If we dont have a validation file provided, create one from the training file.
        if len(self.validation_file) == 0:
            self.validation_file = self.training_file

            # Compute the training / validation ranges based on the data-split and the limiting percentage.
            train_validation_split = self.options.dataset_limit * self.options.train_validation_split
            training_range = (0.0, train_validation_split)
            validation_range = (train_validation_split, self.options.dataset_limit)

        if stage == "fit":
            # Construct primary training datasets
            # Note that only the training dataset should be limited to full events or partial events.
            self.training_dataset = JetReconstructionDataset(
                data_file=self.training_file,
                event_info=event_info_file,
                limit_index=training_range,
                vector_limit=self.options.limit_to_num_jets,
                partial_events=self.options.partial_events,
                randomization_seed=self.options.dataset_randomization
            )
    
            self.validation_dataset = JetReconstructionDataset(
                data_file=self.validation_file,
                event_info=event_info_file,
                limit_index=validation_range,
                vector_limit=self.options.limit_to_num_jets,
                randomization_seed=self.options.dataset_randomization
            )

        # Optionally construct the testing dataset.
        # This is not used in the main training script but is still useful for testing later.
        if stage == 'test':
            self.testing_dataset = None
            if len(self.options.testing_file) > 0:
                self.testing_dataset = JetReconstructionDataset(
                    data_file=self.options.testing_file,
                    event_info=self.options.event_info_file,
                    limit_index=1.0,
                    vector_limit=self.options.limit_to_num_jets
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, drop_last=True, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_options)

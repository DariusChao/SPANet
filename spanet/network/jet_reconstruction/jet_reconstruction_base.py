import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn

from spanet.network.jet_reconstruction.jet_data_module import JetReconstructionDataModule
from spanet.options import Options
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.network.learning_rate_schedules import get_linear_schedule_with_warmup
from spanet.network.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup


class JetReconstructionBase(pl.LightningModule):
    def __init__(self, options: Options):
        super(JetReconstructionBase, self).__init__()

        self.save_hyperparameters(options)
        self.options = options

        self.data_module = JetReconstructionDataModule(options)
        
        # Compute class weights for particles from the training dataset target distribution
        self.balance_particles = False
        if options.balance_particles and options.partial_events:
            index_tensor, weights_tensor = self.data_module.training_dataset.compute_particle_balance()
            self.particle_index_tensor = torch.nn.Parameter(index_tensor, requires_grad=False)
            self.particle_weights_tensor = torch.nn.Parameter(weights_tensor, requires_grad=False)
            self.balance_particles = True

        # Compute class weights for jets from the training dataset target distribution
        self.balance_jets = False
        if options.balance_jets:
            jet_weights_tensor = self.data_module.training_dataset.compute_vector_balance()
            self.jet_weights_tensor = torch.nn.Parameter(jet_weights_tensor, requires_grad=False)
            self.balance_jets = True

        self.balance_classifications = options.balance_classifications
        if self.balance_classifications:
            classification_weights = {
                key: torch.nn.Parameter(value, requires_grad=False)
                for key, value in self.data_module.training_dataset.compute_classification_balance().items()
            }

            self.classification_weights = torch.nn.ParameterDict(classification_weights)

        # Helper arrays for permutation groups. Used for the partial-event loss functions.
        event_permutation_group = np.array(self.event_info.event_permutation_group)
        self.event_permutation_tensor = torch.nn.Parameter(torch.from_numpy(event_permutation_group), False)
        self.event_permutation_list = self.event_permutation_tensor.tolist()

        # Helper variables for keeping track of the number of batches in each epoch.
        # Used for learning rate scheduling and other things.
        self.steps_per_epoch = len(self.data_module.training_dataset) // (self.options.batch_size * max(1, self.options.num_gpu))
        # self.steps_per_epoch = len(self.data_module.training_dataset) // self.options.batch_size
        self.total_steps = self.steps_per_epoch * self.options.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.learning_rate_warmup_epochs))

    @property
    def event_info(self):
        return self.data_module.training_dataset.event_info

    def configure_optimizers(self):
        optimizer = None

        if 'apex' in self.options.optimizer:
            try:
                # noinspection PyUnresolvedReferences
                import apex.optimizers

                if self.options.optimizer == 'apex_adam':
                    optimizer = apex.optimizers.FusedAdam

                elif self.options.optimizer == 'apex_lamb':
                    optimizer = apex.optimizers.FusedLAMB

                else:
                    optimizer = apex.optimizers.FusedSGD

            except ImportError:
                pass

        else:
            optimizer = getattr(torch.optim, self.options.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        decay_mask = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in self.named_parameters()
                           if not any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": self.options.l2_penalty,
            },
            {
                "params": [param for name, param in self.named_parameters()
                           if any(no_decay in name for no_decay in decay_mask)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer(optimizer_grouped_parameters, lr=self.options.learning_rate)

        if self.options.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                 optimizer,
                 num_warmup_steps=self.warmup_steps,
                 num_training_steps=self.total_steps
             )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

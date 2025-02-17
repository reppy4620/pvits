import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from lightning import LightningModule
from torch.utils.data import DataLoader

from x_vits.utils.dataset import ShuffleBatchSampler, batch_by_size
from x_vits.utils.logging import logger


class LitModuleBase(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.automatic_optimization = False

        self.params = params
        self.loss_coef = params.train.loss_coef

        self.frame_segment_size = params.train.frame_segment_size
        self.sample_segment_size = params.train.sample_segment_size
        self.grad_acc_step = params.train.grad_acc_step

        self.net_g = instantiate(params.generator)
        self.net_d = instantiate(params.discriminator)
        logger.info(f"Generator: {sum(p.numel() for p in self.net_g.parameters()) / 1e6:.3f}M")
        logger.info(f"Discriminator: {sum(p.numel() for p in self.net_d.parameters()) / 1e6:.3f}M")

        self.spec_tfm = instantiate(params.mel)
        self.sample_rate = params.mel.sample_rate
        self.hop_length = params.mel.hop_length

        self.collator = instantiate(params.dataset.collator)

        self.valid_save_data = dict()

    def forward(self, inputs):
        # single inference
        if isinstance(inputs[0], str):
            _, phoneme, *_ = inputs
            phonemes = phoneme.unsqueeze(0).to(self.device)
            phone_lengths = torch.tensor([phonemes.size(-1)], device=self.device)
        # batch inference
        else:
            _, phonemes, *_, phone_lengths, _, _, _ = inputs
            phonemes, phone_lengths = phonemes.to(self.device), phone_lengths.to(self.device)
        o, (_, f0, _, _, _) = self.net_g(phonemes, phone_lengths)
        return o.squeeze(1), f0.squeeze(1)

    def _handle_batch(self, batch, batch_idx, train):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self._handle_batch(batch, batch_idx, train=True)

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        mels, wavs, p_attns = self._handle_batch(batch, batch_idx, train=False)

    def on_validation_epoch_end(self):
        logger.info(", ".join(f"{k}={v:.3f}" for k, v in self.trainer.logged_metrics.items()))

    def train_dataloader(self):
        train_ds = instantiate(self.params.dataset.train)
        indices = train_ds.ordered_indices()
        batches = batch_by_size(
            indices=indices,
            num_tokens_fn=train_ds.num_tokens,
            max_tokens=self.params.dataset.max_tokens,
            required_batch_size_multiple=1,
        )
        batch_sampler = ShuffleBatchSampler(batches, drop_last=True, shuffle=True)
        train_dl = DataLoader(
            train_ds,
            num_workers=self.params.train.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
            batch_sampler=batch_sampler,
            persistent_workers=True,
        )
        return train_dl

    def val_dataloader(self):
        val_ds = instantiate(self.params.dataset.valid)
        val_dl = DataLoader(
            val_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.params.train.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )
        return val_dl

    def configure_optimizers(self):
        optimizer_g = instantiate(self.params.optimizer, params=self.net_g.parameters())
        optimizer_d = instantiate(self.params.optimizer, params=self.net_d.parameters())
        scheduler_g = instantiate(self.params.scheduler, optimizer=optimizer_g)
        scheduler_d = instantiate(self.params.scheduler, optimizer=optimizer_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

import torch
import torch.nn.functional as F

from x_vits.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from x_vits.utils.model import slice_segments

from .base import LitModuleBase


class PeriodVITSModule(LitModuleBase):
    def _handle_batch(self, batch, batch_idx, train):
        optimizer_g, optimizer_d = self.optimizers()
        (
            _,  # bname
            x,
            duration,
            spec,
            cf0,
            vuv,
            y,
            x_lengths,
            spec_lengths,
            _,  # y_lengths
            _,
        ) = batch
        y_hat, ids_slice, p_attn, loss_dict = self.net_g.training_step(
            x, x_lengths, spec, spec_lengths, cf0, vuv, duration=duration
        )

        mel = self.spec_tfm.spec_to_mel(spec)
        y_mel = slice_segments(mel, ids_slice, self.params.train.frame_segment_size)
        y_hat_mel = self.spec_tfm.to_mel(y_hat.squeeze(1))

        y = slice_segments(y, ids_slice * self.hop_length, self.params.train.sample_segment_size)

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        with torch.autocast(device_type="cuda", enabled=False):
            loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)

        if train:
            if (batch_idx + 1) % self.grad_acc_step == 0:
                optimizer_d.zero_grad()
            self.manual_backward(loss_disc / self.grad_acc_step)
            if (batch_idx + 1) % self.grad_acc_step == 0:
                optimizer_d.step()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            loss_fm = feature_matching_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_hat_g)
            loss_g = (
                loss_gen
                + self.params.train.loss_coef.mel * loss_mel
                + self.params.train.loss_coef.fm * loss_fm
                + loss_dict["dur"]
                + loss_dict["cf0"]
                + loss_dict["vuv"]
                + self.params.train.loss_coef.kl * loss_dict["kl"]
            )
        if train:
            if (batch_idx + 1) % self.grad_acc_step == 0:
                optimizer_g.zero_grad()
            self.manual_backward(loss_g / self.grad_acc_step)
            if (batch_idx + 1) % self.grad_acc_step == 0:
                optimizer_g.step()

        loss_dict = dict(
            it=self.global_step,
            b=len(x),
            disc=loss_disc,
            gen=loss_gen,
            fm=loss_fm,
            mel=loss_mel,
            **loss_dict,
        )
        self.log_dict(loss_dict, prog_bar=True)
        return y_hat_mel, y_hat, p_attn

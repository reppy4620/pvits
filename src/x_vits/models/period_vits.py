import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from x_vits.losses import kl_loss
from x_vits.modules.handler import DurationHandlerOutput
from x_vits.utils.model import rand_slice_segments, slice_segments, to_log_scale


class PeriodVITS(nn.Module):
    def __init__(
        self,
        text_encoder,
        duration_handler,
        frame_prior_network,
        pitch_predictor,
        flow,
        posterior_encoder,
        vocoder,
        spec_tfm,
        segment_size,
    ):
        super().__init__()

        self.text_encoder = text_encoder
        self.duration_handler = duration_handler
        self.frame_prior_network = frame_prior_network
        self.pitch_predictor = pitch_predictor
        self.flow = flow
        self.posterior_encoder = posterior_encoder
        self.vocoder = vocoder
        self.dec = vocoder

        self.spec_tfm = spec_tfm
        self.segment_size = segment_size

    def training_step(
        self,
        x,
        x_lengths,
        spec,
        spec_lengths,
        cf0,
        vuv,
        duration=None,
        raw_texts=None,
    ):
        mel = self.spec_tfm.spec_to_mel(spec)
        x, x_mask = self.text_encoder(x, x_lengths)

        duration_handler_output: DurationHandlerOutput = self.duration_handler(
            x=x,
            x_mask=x_mask,
            x_lengths=x_lengths,
            mel=mel,
            mel_lengths=spec_lengths,
            duration=duration,
        )
        x_frame, p_attn, _, y_mask, loss_dict = duration_handler_output
        x_frame, m_p, logs_p = self.frame_prior_network(x_frame, y_mask)
        pitch_pred = self.pitch_predictor(x_frame, y_mask)
        log_cf0_pred, vuv_logit_pred = pitch_pred.chunk(2, dim=1)
        vuv_pred = vuv_logit_pred.sigmoid()

        z, m_q, logs_q = self.posterior_encoder(spec, y_mask)
        z_p = self.flow(z, y_mask)

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        f0 = (
            cf0 * vuv
            if random.random() < 0.8
            else (log_cf0_pred.exp() * torch.where(vuv_pred > 0.5, 1.0, 0.0)).detach()
        )
        f0 = slice_segments(f0, ids_slice, self.segment_size)
        o = self.vocoder(z_slice, f0)

        with torch.autocast(device_type="cuda", enabled=False):
            log_cf0 = to_log_scale(cf0)
            loss_cf0 = (
                F.mse_loss(log_cf0_pred * vuv, log_cf0 * vuv, reduction="sum") / spec_lengths.sum()
            )
            loss_vuv = F.mse_loss(vuv_pred, vuv, reduction="sum") / spec_lengths.sum()
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        loss_dict = loss_dict | dict(cf0=loss_cf0, vuv=loss_vuv, kl=loss_kl)
        return o, ids_slice, p_attn, loss_dict

    def forward(self, x, x_lengths, raw_texts, noise_scale=0.667):
        x, x_mask = self.text_encoder(x, x_lengths)

        x_frame, p_attn, duration, y_mask, _ = self.duration_handler.infer(x, x_mask)
        x_frame, m_p, logs_p = self.frame_prior_network(x_frame, y_mask)
        pitch_pred = self.pitch_predictor(x_frame * y_mask, y_mask)
        log_cf0_pred, vuv_logit_pred = pitch_pred.chunk(2, dim=1)
        cf0 = torch.exp(log_cf0_pred)
        vuv = torch.where(vuv_logit_pred.sigmoid() > 0.5, 1.0, 0.0)
        f0 = cf0 * vuv

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow.reverse(z_p * y_mask, y_mask)
        o = self.vocoder(z * y_mask, f0)
        return o, (p_attn, cf0, vuv, duration)

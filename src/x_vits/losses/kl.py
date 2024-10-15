def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * (z_p - m_p).pow(2) * (-2.0 * logs_p).exp()
    loss = (kl * z_mask).sum() / z_mask.sum()
    return loss

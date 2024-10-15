from pprint import pprint

import hydra
import torch
from hydra.utils import instantiate
from lightning import Trainer


def test_model(cfg):
    pprint(cfg)
    model = instantiate(cfg.generator)
    n_spec = 513

    B = 4
    P = 10
    F = 100

    phoneme = torch.randint(1, 30, (B, P))
    duration = torch.randint(1, 5, (B, 1, P), dtype=torch.float)
    spec = torch.randn(B, n_spec, F)
    cf0 = torch.randn(B, 1, F).abs()
    vuv = torch.ones(B, 1, F).float()
    phone_lengths = torch.tensor([P] * 4)
    spec_lengths = torch.tensor([F] * 4)
    raw_texts = ["こんにちは，こんばんは"] * B

    o, o_mb, ids_slice, p_attn, loss_dict = model.training_step(
        phoneme,
        phone_lengths,
        spec,
        spec_lengths,
        cf0,
        vuv,
        duration=duration,
        raw_texts=raw_texts,
    )
    print(o.shape, o_mb.shape)
    pprint(loss_dict)
    print(ids_slice)


def test_train(cfg):
    lit_module = instantiate(cfg.lit_module, params=cfg, _recursive_=False)
    train_dl = lit_module.train_dataloader()
    print(next(iter(train_dl)))
    trainer = Trainer(
        max_steps=1,
        detect_anomaly=True,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model=lit_module)


@torch.no_grad()
@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    print(cfg)
    # test_model(cfg)
    test_train(cfg)


if __name__ == "__main__":
    main()

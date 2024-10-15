from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf
from joblib import Parallel, delayed

from x_vits.frontend.ja import phonemes, pp_symbols
from x_vits.utils.const import PreprocessType
from x_vits.utils.logging import logger
from x_vits.utils.tqdm import tqdm_joblib


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    if PreprocessType.from_str(cfg.preprocess.type) == PreprocessType.JSUT:
        jsut_preprocess(cfg)
    else:
        raise ValueError(f"Invalid preprocess type: {cfg.preprocess.type}")


def jsut_preprocess(cfg):
    if not cfg.preprocess.overwrite and Path(f"{cfg.path.data_root}/done").exists():
        logger.info("Already processed.")
        return

    wav_dir = Path(cfg.path.wav_dir)
    lab_dir = Path(cfg.path.lab_dir)
    transcription_dir = Path(cfg.path.transcription_dir)

    # Create directories
    [
        Path(d).mkdir(parents=True, exist_ok=True)
        for d in [
            cfg.path.crop_wav_dir,
            cfg.path.df_dir,
            cfg.path.cf0_dir,
            cfg.path.vuv_dir,
        ]
    ]

    logger.info("Start Processing...")

    wav_files = list(sorted(wav_dir.glob("*.wav")))

    def _process(wav_file):
        bname = wav_file.stem
        wav, sr = sf.read(wav_file)
        assert sr == cfg.mel.sample_rate
        assert len(wav.shape) == 1

        with open(lab_dir / f"{bname}.lab", "r") as f:
            fullcontext = f.readlines()
            fullcontext = [line.strip() for line in fullcontext]

        start_idx = int(
            (int(fullcontext[0].split()[1]) * 1e-7 - cfg.preprocess.sil_sec) * cfg.mel.sample_rate
        )
        end_idx = int(
            (int(fullcontext[-1].split()[0]) * 1e-7 + cfg.preprocess.sil_sec) * cfg.mel.sample_rate
        )

        wav_clipped = wav[start_idx:end_idx]
        sf.write(f"{cfg.path.crop_wav_dir}/{bname}.wav", wav_clipped, sr)

        f0, _ = pw.harvest(
            wav_clipped, sr, frame_period=cfg.mel.hop_length / cfg.mel.sample_rate * 1e3
        )
        vuv = (f0 != 0).astype(np.float32)
        x = np.arange(len(f0))
        idx = np.nonzero(f0)
        cf0 = np.interp(x, x[idx], f0[idx])
        np.save(f"{cfg.path.cf0_dir}/{bname}.npy", cf0)
        np.save(f"{cfg.path.vuv_dir}/{bname}.npy", vuv)

        frame_length = len(wav_clipped) // cfg.mel.hop_length
        label = pp_symbols(fullcontext)
        durations = []
        cnt = 0
        for s in label:
            if s in phonemes or s in ["^", "$", "?", "_"]:
                s, e, _ = fullcontext[cnt].split()
                s, e = int(s), int(e)
                dur = (e - s) * 1e-7 / (cfg.mel.hop_length / cfg.mel.sample_rate)
                durations.append(dur)
                cnt += 1
            else:
                durations.append(1)
        durations[0] = int(cfg.preprocess.sil_sec * cfg.mel.sample_rate / cfg.mel.hop_length)
        durations[-1] = int(cfg.preprocess.sil_sec * cfg.mel.sample_rate / cfg.mel.hop_length)
        # adjust length, differences are caused by round op.
        round_durations = np.round(durations)
        diff_length = np.sum(round_durations) - frame_length
        if diff_length != 0:
            if diff_length > 0:
                durations_diff = round_durations - durations
                d = -1
            else:  # diff_length < 0
                durations_diff = durations - round_durations
                d = 1
            sort_dur_idx = np.argsort(durations_diff)[::-1]
            for i, idx in enumerate(sort_dur_idx, start=1):
                round_durations[idx] += d
                if i == abs(diff_length):
                    break
            assert np.sum(round_durations) == frame_length
        label = " ".join(label)
        duration = " ".join([str(int(d)) for d in round_durations])

        with open(transcription_dir / f"{bname}.txt", "r") as f:
            raw_text = f.readline().strip()
        return bname, label, duration, frame_length, raw_text

    with tqdm_joblib(len(wav_files)):
        out = Parallel(n_jobs=cfg.preprocess.n_jobs)(delayed(_process)(f) for f in wav_files)

    assert len(out) == len(wav_files)
    valid_size = int(len(wav_files) * 0.02)
    df = pd.DataFrame(
        list(sorted(out, key=lambda x: x[0])),
        columns=["bname", "label", "duration", "frame_length", "raw_text"],
    )
    df.to_csv(f"{cfg.path.df_dir}/all.csv", index=False)
    train_df = df.iloc[valid_size:]
    valid_df = df.iloc[:valid_size]
    train_df.to_csv(cfg.path.train_df_file, index=False)
    valid_df.to_csv(cfg.path.valid_df_file, index=False)

    with open(f"{cfg.path.data_root}/done", "w") as f:
        f.write("done")


if __name__ == "__main__":
    main()

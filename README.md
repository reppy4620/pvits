x-vits
===

This repo contains experiments for confirming performance of PeriodVITS.

Main model is some modification version of PeriodVITS.
- PeriodVITS
    - +deberta hidden representations to text encoder
    - +style encoder with style diffusion like StyleTTS2
    - +multi-stream bigvgan with bigvgan-v1 discriminator


Now only supports JSUT corpus.


# Citations
```
@inproceedings{kim2021conditional,
  title={Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={International Conference on Machine Learning},
  pages={5530--5540},
  year={2021},
  organization={PMLR}
}
@inproceedings{shirahata2023period,
  title={Period vits: Variational inference with explicit pitch modeling for end-to-end emotional speech synthesis},
  author={Shirahata, Yuma and Yamamoto, Ryuichi and Song, Eunwoo and Terashima, Ryo and Kim, Jae-Min and Tachibana, Kentaro},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
@inproceedings{li2023styletts,
    title={Style{TTS} 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models},
    author={Yinghao Aaron Li and Cong Han and Vinay S Raghavan and Gavin Mischler and Nima Mesgarani},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=m0RbqrUM26}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
```
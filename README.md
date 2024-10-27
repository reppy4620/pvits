x-vits
===

This repository contains experiments to confirm the performance of PeriodVITS, but not limited to it.  
The focus of this repository is on small and high-quality models that can be trained on a single consumer GPU, such as the RTX3090 or 4090.  
Now only supports JSUT and LJSpeech corpus.  

The model is some modification version of PeriodVITS.
- PeriodVITS
    - +roformer(like llama3) text encoder
    - +deberta-v3-xsmall hidden representations added to text encoder incorporated with cross attention
    - +style encoder with style diffusion(for predicting style vector in inference time) like StyleTTS2 but not using AdaLN now
    - +multi-band bigvgan with bigvgan-v1 discriminator


# Supoort Model
- PeriodVITS
- PeriodVITS with DeBERTa-v3-xsmall hidden representations aggregated by LSTM which is simply added to phoneme embedding.
- X-VITS : as explained above.

#!/bin/bash

bin_dir=../../../../src/x_vits/bin

model_name=period_vits

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python ${bin_dir}/train.py \
    generator=${model_name} \
    generator.text_encoder.num_vocab=151 \
    lit_module=${model_name} \
    dataset=single_en \
    path=ljspeech \
    mel=ljspeech

#!/bin/bash

bin_dir=../../../../src/x_vits/bin

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python ${bin_dir}/synthesize.py \
    path=ljspeech \
    dataset=single_en \
    mel=ljspeech

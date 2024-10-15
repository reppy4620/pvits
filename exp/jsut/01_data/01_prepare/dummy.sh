#!/bin/bash

out_dir=/path/to/data_root
mkdir -p ${out_dir}/{transcription,wav,lab}

# <bname>:<transcription>
metadata_file=/path/to/jsut_ver1.1/basic5000/transcript_utf8.txt

while IFS=: read -r bname transcription
do
    echo ${bname}
    echo "$transcription" > "${out_dir}/transcription/${bname}.txt"
done < ${metadata_file}

wav_dir=/path/to/jsut_ver1.1/basic5000/wav_24k
lab_dir=/path/to/jsut-label/labels/basic5000

rsync -avu ${wav_dir}/ ${out_dir}/wav/
rsync -avu ${lab_dir}/ ${out_dir}/lab/

# cd ${out_dir}/wav
# for fpath in ${wav_dir}/*.wav; do
#     bname=$(basename ${fname} .wav)
#     echo ${bname}
#     ln -s ${fpath} ${bname}.wav
# done
# cd -

# cd ${out_dir}/lab
# for fpath in ${lab_dir}/*.lab; do
#     bname=$(basename ${fname} .lab)
#     echo ${bname}
#     ln -s ${fpath} ${bname}.lab
# done

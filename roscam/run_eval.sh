#!/bin/bash
while true; do
    export CUDA_VISIBLE_DEVICES=0 
    MODEL="$HOME/models/grefexp_gen_mao.h5"
    CMD="python evaluation/evaluate_image_captioning_mao.py --model $MODEL --count 100"
    $CMD | tee /tmp/eval_output_tmp.txt
    mv /tmp/eval_output_tmp.txt $HOME/public/latest_results.txt
    rm /tmp/eval_output_tmp.txt
    sleep 1
done

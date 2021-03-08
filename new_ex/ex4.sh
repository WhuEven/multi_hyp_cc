#!/bin/bash

#在NUS数据集上的结果，用Gehler-Shi和Cube+预训练


# Declare an array of string with type
declare -a CAMERAS=("canon_eos_1D_mark3" "canon_eos_600D" "fuji" "nikonD5200" "panasonic" "olympus" "sony" "samsung" )

TRAINED_MODEL_PATH="./new_ex_output/ex4"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py new_ex/ex4_step1.json Multidataset cube+_shigehler data/multidataset/cube+_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

# Iterate the string array using for loop
for camera in ${CAMERAS[@]}; do
   CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py new_ex/ex4_step2.json data/nus/$camera.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH/$camera/ --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/cube+_shigehler/table2_pre_pretrain/0/model_best.pth.tar
done
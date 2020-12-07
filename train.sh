#!/bin/bash
runModel(){
    CONFIG_FILE=configs/$1.py
    WORK_DIR=result/$2
    rm -rf ${WORK_DIR}
    python linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
}
pdbModel(){
    CONFIG_FILE=configs/$1.py
    WORK_DIR=result/$2
    python -m pdb linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
}
testModel(){
    CONFIG_FILE=configs/$1.py
    CKPT=result/$2/epoch_end.pth
    rm -rf ${WORK_DIR}
    python linear_val.py --config ${CONFIG_FILE} --ckpt ${CKPT}
}
export CUDA_VISIBLE_DEVICES=0

runModel resnet50_imagenet resnet50_PWS

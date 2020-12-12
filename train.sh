#!/bin/bash
runModel(){
    CONFIG_FILE=configs/${DATASET}/$1.py
    WORK_DIR=result/$2
    rm -rf ${WORK_DIR}
    python linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
}
pdbModel(){
    CONFIG_FILE=configs/$1.py
    WORK_DIR=result/$2
    python -m pdb linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
}

export CUDA_VISIBLE_DEVICES=1

DATASET=cifar
runModel pws pws
#!/bin/bash

# workdir=$(cd "${dirname "$0"}" || exit; pwd)
workdir=$(cd $(dirname $0); pwd)


cd ${workdir}

CUDA_VISIBLE_DEVICES=0,1,2,3 python detect.py
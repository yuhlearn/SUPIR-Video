#! /bin/bash

CONDA_BASE_DIR="$(conda info | sed -n -e 's/^.*base environment : //p' | cut -d " " -f1)"
source "$CONDA_BASE_DIR/bin/deactivate" && source "$CONDA_BASE_DIR/bin/activate" SUPIR

python webui.py --loading_half_params --use_fp8_unet #--use_fp8_vae

#!/bin/bash 

# To enable using canda
source /home/hayato.imafuku/miniconda3/etc/profile.d/conda.sh 

conda activate env

# Set environment variables to limit the number of threads for parallel processing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run the Python script 
python run_H0_estimation_for_pp_plot.py

# Deactivate environment 
conda deactivate
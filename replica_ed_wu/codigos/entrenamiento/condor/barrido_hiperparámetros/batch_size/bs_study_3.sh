#!/bin/bash
cd /lhome/ext/i3m121/i3m1214/replica_ed_wu/codigos/entrenamiento/condor/barrido_hiperparámetros/
source /lhome/ext/i3m121/i3m1214/replica_ed_wu/codigos/entrenamiento/condor/barrido_hiperparámetros/setup.sh
python totrain.py --lr 0.00001 --batch_size 4 --num_epochs 20 --wandb_project 'bs_study' --weight_decay 0.01
#!/bin/bash
cd /lhome/ext/i3m121/i3m1213/barrido/barrido_hiperparámetros/
source /lhome/ext/i3m121/i3m1213/barrido/barrido_hiperparámetros/setup.sh
python totrain.py --lr 0.00001 --batch_size 4 --num_epochs 125 --wandb_project 'definitivo1_study' --weight_decay 0.01

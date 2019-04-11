#!/usr/bin/env bash

ptyhon testSetting.py --state $1

echo 'preparing training data'
python analyzeData.py --mode train
echo 'merging ......'
python mergeData.py
echo 'generating historical data ......'
python generateHistoricData.py

echo 'training ......'

python train_model_1228.py --mode train --epoch 3720 &
python train_model_1294.py --mode train --epoch 920 &
python train_model_1305.py --mode train --epoch 1420 &
python train_model_1228_tol.py --mode train --epoch 3720 &
python train_model_1294_tol.py --mode train --epoch 920 &
python train_model_1305_tol.py --mode train --epoch 1420 &
wait

python train_model_1250.py --mode train --epoch 5060 &
python train_model_1266.py --mode train --epoch 8200 &
python train_model_1250_tol.py --mode train --epoch 5060 &
python train_model_1266_tol.py --mode train --epoch 8200 &
wait



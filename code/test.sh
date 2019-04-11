#!/usr/bin/env bash

python testSetting.py --state $1

echo 'preparing test data'
python analyzeData.py --mode pred
echo 'generating historical features'
python generateTestFeatures.py

echo 'clean dir pred'
python cleanDir.py  --which pred

# pred
echo 'predicting ......'

python train_model_1228.py --mode pred --epoch 1
python train_model_1250.py --mode pred --epoch 1
python train_model_1266.py --mode pred --epoch 1
python train_model_1294.py --mode pred --epoch 1
python train_model_1305.py --mode pred --epoch 1
wait

python train_model_1228_tol.py --mode pred --epoch 1
python train_model_1250_tol.py --mode pred --epoch 1
python train_model_1266_tol.py --mode pred --epoch 1
python train_model_1294_tol.py --mode pred --epoch 1
python train_model_1305_tol.py --mode pred --epoch 1
wait

echo 'blending ......'

python blend_results.py

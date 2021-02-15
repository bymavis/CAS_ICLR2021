# Improving Adversarial Robustness via Channel-wise Activation Suppressing

This is our Pytorch implementation of CAS.

## Pre-trained model

Pre-trained models of AT+CAS, Trades+CAS, MART+CAS on cifar10 are in the folder of ./CAS_checkpoint

## Training

1. Train AT+CAS

   ```
   python ./Standard_Adv_Training/train.py --gpu 0 --adv_train --affix AT_CAS
   ```

2. Train Trades+CAS

   ```
   python ./Trades/train_trades_cifar10_cas.py
   ```

3. Train MART+CAS

   ```
   python ./MART/train_mart_cas.py
   ```

## Test

Specify the checkpoint path as argument

```
python ./Trades/pgd_attack_cifar10_cas.py --model_path xxx
```


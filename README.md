# Improving Adversarial Robustness via Channel-wise Activation Suppressing (ICLR2021)

This is the official Pytorch implementation of CAS.


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


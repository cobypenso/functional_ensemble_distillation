# CIFAR Baselines

This project implements existing baseline models on CIFAR dataset.
Try to report the best performance for each baseline, although it involves cumbersome tuning of hyperparameters.

## Setup

```
ln -s ../../giung2/ ./
ln -s ../../datasets/ ./
```

## Standard Baselines

### WRN28x1-BN-ReLU on CIFAR-10

> All models are trained on the first 45,000 examples of the train split of CIFAR-10; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.9231 | 0.2801 | 0.0396 | 0.1031 | -      | 0.2346 | 0.0043 | 0.2476 | 1.6781 |
| DE-2       | 2     | 0.9423 | 0.1997 | 0.0141 | 0.1375 | 0.2619 | 0.1917 | 0.0160 | 0.1978 | 1.3086 |
| DE-4       | 4     | 0.9460 | 0.1701 | 0.0086 | 0.1560 | 0.2528 | 0.1697 | 0.0119 | 0.1769 | 1.1031 |
| DE-8       | 8     | 0.9507 | 0.1543 | 0.0118 | 0.1649 | 0.2440 | 0.1543 | 0.0119 | 0.1653 | 1.0016 |
| Dropout    | 1     | 0.9356 | 0.2529 | 0.0341 | 0.0845 | -      | 0.2086 | 0.0061 | 0.2099 | 1.6383 |
| MC-Dropout | 30    | 0.9361 | 0.2215 | 0.0226 | 0.1127 | 0.0539 | 0.2012 | 0.0055 | 0.2012 | 1.4227 |
| BE-4       | 4     | 0.9318 | 0.2949 | 0.0413 | 0.0704 | 0.0192 | 0.2077 | 0.0060 | 0.1982 | 2.0969 |
| MIMO-2     | 2     | 0.9155 | 0.2601 | 0.0098 | 0.2547 | 0.2718 | 0.2601 | 0.0102 | 0.2572 | 1.0070 |
| DUQ        | 1     | 0.9284 | 0.2950 | 0.0359 | 0.1031 | 0.0000 | 0.2535 | 0.0082 | 0.2265 | 1.4297 |

### WRN28x1-BN-ReLU on CIFAR-100

> All models are trained on the first 45,000 examples of the train split of CIFAR-100; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method     | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-         | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD        | 1     | 0.6917 | 1.2164 | 0.1124 | 0.6259 | -      | 1.1002 | 0.0125 | 1.1395 | 1.5102 |
| DE-2       | 2     | 0.7247 | 1.0003 | 0.0266 | 0.7747 | 1.0543 | 0.9797 | 0.0318 | 0.9920 | 1.2102 |
| DE-4       | 4     | 0.7499 | 0.8928 | 0.0279 | 0.8834 | 1.0527 | 0.8928 | 0.0316 | 0.9007 | 1.0156 |
| DE-8       | 8     | 0.7663 | 0.8357 | 0.0528 | 0.9454 | 1.0395 | 0.8301 | 0.0261 | 0.8326 | 0.9039 |

### WRN28x10-BN-ReLU on CIFAR-100

> All models are trained on the first 45,000 examples of the train split of CIFAR-100; the last 5,000 examples of the train split are used as the validation split. We basically follow the standard data augmentation policy which consists of random cropping of 32 pixels with a padding of 4 pixels and random horizontal flipping.

| Method            | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-                | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| SGD               | 1     | 0.8050 | 0.8007 | 0.0584 | 0.6338 | -      | 0.7922 | 0.0398 | 0.8383 | 1.1453 |
| DE-2              | 2     | 0.8168 | 0.7091 | 0.0260 | 0.7254 | 0.3456 | 0.7093 | 0.0252 | 0.7369 | 1.0078 |
| DE-4              | 4     | 0.8248 | 0.6599 | 0.0213 | 0.7614 | 0.3557 | 0.6564 | 0.0238 | 0.6784 | 0.9445 |
| DE-8              | 8     | 0.8303 | 0.6302 | 0.0217 | 0.7784 | 0.3537 | 0.6224 | 0.0217 | 0.6432 | 0.9102 |
| Dropout           | 1     | 0.8028 | 0.8226 | 0.0583 | 0.6500 | -      | 0.8131 | 0.0448 | 0.8478 | 1.1430 |
| MC-Dropout        | 30    | 0.8024 | 0.8087 | 0.0523 | 0.6785 | 0.0299 | 0.8034 | 0.0421 | 0.8359 | 1.1109 |
| SpatialDropout    | 1     | 0.8035 | 0.8007 | 0.0640 | 0.5761 | -      | 0.7831 | 0.0441 | 0.7968 | 1.1758 |
| MC-SpatialDropout | 30    | 0.8041 | 0.7685 | 0.0503 | 0.6381 | 0.0735 | 0.7627 | 0.0387 | 0.7780 | 1.1039 |
| BE-4              | 4     | 0.8114 | 0.7872 | 0.0641 | 0.4615 | 0.0939 | 0.7444 | 0.0318 | 0.7811 | 1.3266 |
| MIMO-3            | 3     | 0.8101 | 0.7182 | 0.0193 | 0.6836 | 0.5602 | 0.7181 | 0.0209 | 0.7478 | 1.0438 |

## Baselines for Bayesian Interpretation

### R20-FRN-SiLU

> All models are trained on the first 40,960 examples of the train split of CIFAR-10-HMC; the last 9,040 examples of the train split are used as the validation split. For a clear Bayesian interpretation of the inference procedure, (1) we do not use any data augmentation, and (2) batch normalization is replaced with filter response normalization.

| Method           | # Ens | ACC    | NLL    | ECE    | ENT    | KLD    | NLL-TS | ECE-TS | ENT-TS | TS     |
| :-               | :-:   | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| HMC<sup>1</sup>  | 30    | 0.8998 | 0.3222 | 0.0423 | 0.3919 | 1.0216 | 0.3160 | 0.0103 | 0.2941 | 0.7719 |
|                  | 120   | 0.9026 | 0.3114 | 0.0488 | 0.4148 | N/A    | 0.2980 | 0.0109 | 0.2750 | 0.7109 |
|                  | 720   | 0.9071 | 0.3067 | 0.0592 | 0.4418 | N/A    | 0.2841 | 0.0057 | 0.2669 | 0.6727 |
| SGD              | 1     | 0.8477 | 0.5855 | 0.0899 | 0.1980 | -      | 0.4974 | 0.0296 | 0.4876 | 1.5203 |
| DE-2             | 2     | 0.8670 | 0.4446 | 0.0329 | 0.2734 | 0.4830 | 0.4212 | 0.0180 | 0.4115 | 1.2820 |
| DE-4             | 4     | 0.8822 | 0.3762 | 0.0175 | 0.3206 | 0.4848 | 0.3735 | 0.0195 | 0.3582 | 1.0797 |
| DE-8             | 8     | 0.8882 | 0.3445 | 0.0169 | 0.3475 | 0.4851 | 0.3446 | 0.0150 | 0.3348 | 0.9734 |
| DE-16            | 16    | 0.8917 | 0.3276 | 0.0157 | 0.3632 | N/A    | 0.3263 | 0.0099 | 0.3204 | 0.9133 |
| DE-64            | 64    | 0.8920 | 0.3186 | 0.0186 | 0.3785 | N/A    | 0.3149 | 0.0055 | 0.3030 | 0.8539 |
| DE-256           | 256   | 0.8943 | 0.3143 | 0.0226 | 0.3843 | N/A    | 0.3093 | 0.0068 | 0.2982 | 0.8375 |
| DE-512           | 512   | 0.8937 | 0.3147 | 0.0216 | 0.3854 | N/A    | 0.3096 | 0.0072 | 0.2981 | 0.8359 |
| Dropout          | 1     | 0.8709 | 0.4567 | 0.0709 | 0.1754 | -      | 0.3956 | 0.0221 | 0.4003 | 1.5109 |
| MC-Dropout       | 30    | 0.8723 | 0.3803 | 0.0098 | 0.3699 | 0.3293 | 0.3802 | 0.0089 | 0.3797 | 1.0187 |

* * *

<sup>1</sup>
Here, we use HMC samples from [Izmailov et al. (2021)](https://arxiv.org/abs/2104.14421).

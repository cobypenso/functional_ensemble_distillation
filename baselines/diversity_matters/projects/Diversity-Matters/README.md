# Diversity Matters When Learning From Ensembles

This repository is the official implementation of [Diversity Matters When Learning From Ensembles]() (NeurIPS 2021).

## Training

To train DE-4 teachers with WRN28x10 on CIFAR-100, run the following commands:
```
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_0/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_1/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_2/
python ./scripts/train_teacher.py --config-file ./configs/C100_WRN28x10_SGD.yaml OUTPUT_DIR ./outputs/C100_WRN28x10_SGD_3/
```

To train BE-4 students with WRN28x10 on CIFAR-100, run the following commands:
```
python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 1.0 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KD_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 1.0 --kd-method-name gaussian \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDGaussian_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 1.0 --kd-method-name ods_l2 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDODS_0/

python ./scripts/train_student.py --config-file ./configs/C100_WRN28x10_BE4.yaml \
                                  --kd-teacher-config-file ./configs/C100_WRN28x10_SGD.yaml \
                                  --kd-teacher-weight-file ./outputs/C100_WRN28x10_SGD_0/best_acc1.pth.tar \
                                  --kd-alpha 0.9 --kd-temperature 1.0 --kd-method-name c_ods_l2 \
                                  OUTPUT_DIR ./outputs/C100_WRN28x10_BE4_KDConfODS_0/
```

## Evaluation

[`./scripts/evaluation.ipynb`](./scripts/evaluation.ipynb) includes evaluation of WRN28x10 on CIFAR-100, and here are the results:
```
Label                         ACC    NLL     BS    ECE    cNLL    cBS    cECE
--------------------------  -----  -----  -----  -----  ------  -----  ------
DeepEns-1                   80.22  0.789  0.282  0.042   0.789  0.282   0.041
DeepEns-2                   81.90  0.713  0.261  0.033   0.708  0.260   0.031
DeepEns-3                   82.46  0.684  0.253  0.032   0.673  0.251   0.027
DeepEns-4                   82.54  0.670  0.249  0.033   0.655  0.246   0.026
BatchEns-4 (KD)             80.40  0.804  0.286  0.072   0.750  0.277   0.021
BatchEns-4 (KD + Gaussian)  80.04  0.816  0.288  0.075   0.760  0.277   0.020
BatchEns-4 (KD + ODS)       81.92  0.685  0.258  0.026   0.682  0.258   0.026
BatchEns-4 (KD + ConfODS)   82.25  0.670  0.253  0.023   0.665  0.252   0.023
```

## Citation

If you find this useful in your research, please consider citing our paper:
```
@inproceedings{nam2021diversity,
  title     = {Diversity Matters When Learning From Ensembles},
  author    = {Giung Nam and Jongmin Yoon and Yoonho Lee and Juho Lee},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}
```

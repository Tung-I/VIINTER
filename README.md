# VIINTER


## Environment
- cuda 11.8
- python 3.10

1. Create a new conda env:
```shell
conda create -n viinter python=3.10
source activate viinter
```

2. Install the packages by:
```shell
./install_env.sh 
```

## Training
To train the first dataset:
```shell
./train_dynerf.sh PATH-TO-DATASET/dataset_for_A100
```

To train the second:
```shell
./train_mipnerf.sh PATH-TO-DATASET/dataset_for_A100
```

And, the last:
```shell
./train_llff.sh PATH-TO-DATASET/dataset_for_A100
```

The checkpoints should be saved in ./exps
# Deep Bilateral Learning for Real-Time Image Enhancements
This repository contains a PyTorch reimplementation of [Deep Bilateral Learning for Real-Time Image Enhancements](https://groups.csail.mit.edu/graphics/hdrnet/).

## Dataset

An small example "false color" dataset is available [here](https://drive.google.com/file/d/1Gq2fzDTxogsR9KXOLYUlaVuMIXpHgHAI/view?usp=sharing). Note that raw images in .dng format are also supported.

A dataset folder should have the following structure:

```bash
dataset
├── test
│   ├── input
│   └── output
└── train
    ├── input
    └── output
```

## Usage
To train a model (on GPU), run the following command:
```bash
python train.py --epochs=1000 --data_dir=<data_dir> --eval_data_dir=<eval_data_dir> --cuda
```

To test a model on a single image, run the following command:
```bash
python test.py --ckpt_path=<ckpt_path> --test_path=<test_path> --cuda
```

## Known issues and limitations
* HDRnet is a lightweight model, but it requires high resolution images for training, so data processing could be a bottleneck for speed. For generality, we did not do much optimisation in this regard, so the current data pipeline is quite slow. One possible solution is to rewrite the dataset classes to pre-load and pre-process all images before training.
* You could switch on data augmentation in `transforms.Compose`, but the for the same reason as above, it is very inefficient to do data augmentation on CPU. One possible solution is to do data augmentation on GPU.
* Currently, the model has a fixed structure as decribed in the original paper. This implementation does not support a variable network structure like the [official ten TensorFlow implementation](https://github.com/google/hdrnet.git).

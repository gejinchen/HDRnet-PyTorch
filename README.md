# Deep Bilateral Learning for Real-Time Image Enhancements
This repository contains a PyTorch reimplementation of [Deep Bilateral Learning for Real-Time Image Enhancements](https://groups.csail.mit.edu/graphics/hdrnet/).
<p float="left">
  <img src="assets/false_in.jpg" width="250" />
  <img src="assets/false_out.jpg" width="250" /> 
  <img src="assets/false_gt.jpg" width="250" />
</p>
Input / Output / Ground Truth

## Dataset

This implementation supports raw images in .dng format and LDR images. A dataset folder ideally should have the following structure:

```bash
dataset
├── train
│   ├── input
│   └── output
└── eval
    ├── input
    └── output
```
A small example "false color" dataset is available [here](https://drive.google.com/file/d/1Aix4Snl4mwBjGxkn_irqehcmfayMG8XO/view?usp=sharing).

## Usage
To train a model (on GPU), run the following command:
```bash
python train.py --epochs=1000 --train_data_dir=<train_data_dir> --eval_data_dir=<eval_data_dir> --cuda
```

To test a model on a single image, run the following command:
```bash
python test.py --ckpt_path=<ckpt_path> --test_img_path=<test_img_path> --cuda
```

## Known issues and limitations
* HDRnet is a lightweight model, but it requires high-resolution images for training, so data processing could be a bottleneck for speed. For generality, we did not do much optimisation in this regard, so the current data pipeline is quite slow. One possible solution is to rewrite the dataset classes to pre-load and pre-process all images before training.
* You could switch on data augmentation in `transforms.Compose`, but for the same reason as above, it is very inefficient to do data augmentation on CPU. One possible solution is to do data augmentation on GPU.
* Currently, the model has a fixed structure as described in the original paper. This implementation does not support a variable network structure like the [official TensorFlow implementation](https://github.com/google/hdrnet.git).

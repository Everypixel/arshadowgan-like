# GAN training on shadow generation task example
![Alt text](imgs/demo-en.png?raw=true "Title")
### Colab Notebook
PyTorch Colab notebook: <a href="https://colab.research.google.com/drive/1fZl1Pb-qWa6OZQgJ-9SMBm8Dd_WY1fsq?usp=sharing">ARShadowGAN-like</a>
### Prerequisites
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN
### Getting Started
#### Installation
* Clone this repo:
```bash
git clone https://github.com/Everypixel/arshadowgan-like.git
cd arshadowgan
```
* Install dependencies (e.g., segmentation_models_pytorch, ...)
```bash
pip install -r requirements.txt
```
#### Dataset preparation
##### ARShadow-dataset
We will use the <a href="https://drive.google.com/file/d/1CsKIg8tV6gP35l_u3Dg-RKrXBggJrNaL/view?usp=sharing">shadow-ar dataset</a> for training and testing our model. We have already splitted it to train and test parts. Download and extract it please .

##### Your own dataset
Your own dataset has to have the structure such ShadowAR-dataset has. Each folder contains images.

<pre>
dataset
├── train
│   ├── noshadow ── example1.png, ...
│   ├── shadow ──── example1.png, ...
│   ├── mask ────── example1.png, ...
│   ├── robject ─── example1.png, ...
│   └── rshadow ─── example1.png, ...
└── test
    ├── noshadow ── example2.png, ...
    ├── shadow ──── example2.png, ...
    ├── mask ────── example2.png, ...
    ├── robject ─── example2.png, ...
    └── rshadow ─── example2.png, ...
</pre>
* *noshadow* - no shadow images
* *shadow* - images with shadow
* *mask* - inserted object masks
* *robject* - occluders masks
* *rshadow* - occluders shadows
### Training
#### Training attention module
Set the parameters:
* *dataset_path* - path to dataset
* *model_path* - path for attention model saving
* *batch_size* - amount of images in batch <br>(reduce it if "CUDA: out of memory" error)
* *seed* - seed for random functions
* *img_size* - image width or image height (is divisible by 32)
* *lr* - learning rate
* *n_epoch* - amount of epochs<br><br>

For example:
``` bash
python3 scripts/train_attention.py \
       --dataset_path '/content/arshadowgan/dataset/' \
       --model_path '/content/drive/MyDrive/attention128.pth' \
       --batch_size 200 \
       --seed 42 \
       --img_size 256 \
       --lr 1e-4 \
       --n_epoch 100
```
#### Training shadow-generation module

* *dataset_path* - path to dataset
* *Gmodel_path* - path for generator model saving
* *Dmodel_path* - path for discriminator model saving
* *batch_size* - amount of images in batch <br>(reduce it if "CUDA: out of memory" error)
* *seed* - seed for random functions
* *img_size* - image width or image height (is divisible by 32)
* *lr_G* - generator learning rate
* *lr_D* - discriminator learning rate
* *n_epoch* - amount of epochs
* *betta1,2,3* - loss function coefficients, see <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_ARShadowGAN_Shadow_Generative_Adversarial_Network_for_Augmented_Reality_in_Single_CVPR_2020_paper.html">ARShadowGAN</a> paper<br><br>

For example:
``` bash
python3 scripts/train_SG.py \
       --dataset_path '/content/arshadowgan/dataset/' \
       --Gmodel_path '/content/drive/MyDrive/SG_generator.pth' \
       --Dmodel_path '/content/drive/MyDrive/SG_discriminator.pth' \
       --batch_size 64 \
       --seed 42 \
       --img_size 256 \
       --lr_G 1e-4 \
       --lr_D 1e-6 \
       --n_epoch 600 \
       --betta1 10 \
       --betta2 1 \
       --betta3 1e-2 \
       --patience 10 \
       --encoder 'resnet18'
```
### Run
Start inference with results saving<br><br>

For example:
``` bash
python3 scripts/test.py \
       --batch_size 1 \
       --img_size 256 \
       --dataset_path '/content/arshadowgan/dataset/test' \
       --result_path '/content/arshadowgan/results' \
       --path_att '/content/drive/MyDrive/ARShadowGAN-like/attention.pth' \
       --path_SG '/content/drive/MyDrive/ARShadowGAN-like/SG_generator.pth'
```
### Acknowledgements

We thank <a href="https://github.com/ldq9526/ARShadowGAN">ARShadowGAN</a> authors for their amazing work.<br>
We also thank <a href="https://github.com/qubvel/segmentation_models.pytorch">segmentation_models.pytorch</a> for network architecture, <a href="https://github.com/albumentations-team/albumentations">albumentations</a> for augmentations, <a href="https://github.com/eriklindernoren/PyTorch-GAN">PyTorch-GAN</a> for discriminator architecture and <a href="https://github.com/photosynthesis-team/piq">piq</a> for Content loss.

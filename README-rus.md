# Обучение GAN на примере ARShadowGAN-like архитектуры
![Alt text](imgs/demo-rus.png?raw=true "Title")
### Colab Notebook
PyTorch Colab notebook: <a href="https://colab.research.google.com/drive/159bHQiaVhs8t5J_3DqwFrlgTF2YTHrbO?usp=sharing">ARShadowGAN-like</a>
### Зависимости
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN
### С чего начать
#### Установка
* Склонируйте этот репозиторий:
```bash
git clone https://github.com/Everypixel/arshadowgan-like.git
cd arshadowgan
```
* Установите требуемые модули (например, segmentation_models_pytorch, ...)
```bash
pip install -r requirements.txt
```
#### Подготовьте датасет
##### ARShadow-датасет
Для обучения и тестирования будем использовать <a href="https://drive.google.com/file/d/1CsKIg8tV6gP35l_u3Dg-RKrXBggJrNaL/view?usp=sharing">готовый датасет</a>.
В нём данные уже разбиты на train и test выборки. Скачайте и распакуйте его.

##### Свой датасет
Ваш датасет должен иметь структуру, похожую на ShadowAR-dataset. В каждой из папок - изображения.

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
* *noshadow* - изображения без теней
* *shadow* - изображения с тенями
* *mask* - маски вставленного объекта
* *robject* - маски соседних объектов (маски окклюдеров)
* *rshadow* - маски теней от соседних объектов
### Обучение
#### Запустим обучение attention-блока
Зададим параметры:
* *dataset_path* - путь до датасета
* *model_path* - путь для сохранения attention-block модели
* *batch_size* - число изображений, прогоняемых через сеть за один раз <br>(число нужно уменьшить в случае появления ошибки CUDA: out of memory)
* *seed* - зерно рандома
* *img_size* - размер изображения (число, кратное 32)
* *lr* - скорость обучения
* *n_epoch* - число эпох<br><br>

Например:
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
#### Запустим обучение shadow-generation блока

* *dataset_path* - путь до датасета
* *Gmodel_path* - путь для сохранения модели генератора
* *Dmodel_path* - путь для сохранения модели дискриминатора
* *batch_size* - число изображений, прогоняемых через сеть за один раз <br>(число нужно уменьшить в случае появления ошибки CUDA: out of memory)
* *seed* - зерно рандома
* *img_size* - размер изображения (число, кратное 2^5)
* *lr_G* - скорость обучения генератора
* *lr_D* - скорость обучения дискриминатора
* *n_epoch* - число эпох
* *betta1,2,3* - коэффициенты функций потерь, см. статью ARShadowGAN<br><br>

Например:
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
### Инференс
Запуск тестирования с сохранением результатов.<br><br>
Например:
``` bash
python3 scripts/test.py \
       --batch_size 1 \
       --img_size 256 \
       --dataset_path '/content/arshadowgan/dataset/test' \
       --result_path '/content/arshadowgan/results' \
       --path_att '/content/drive/MyDrive/ARShadowGAN-like/attention.pth' \
       --path_SG '/content/drive/MyDrive/ARShadowGAN-like/SG_generator.pth'
```
### Благодарности
Мы благодарим авторов статьи <a href="https://github.com/ldq9526/ARShadowGAN">ARShadowGAN</a>.<br>
Также выражаем благодарность <a href="https://github.com/qubvel/segmentation_models.pytorch">segmentation_models.pytorch</a> за архитектуру сети, <a href="https://github.com/albumentations-team/albumentations">albumentations</a> за аугментации, <a href="https://github.com/eriklindernoren/PyTorch-GAN">PyTorch-GAN</a> за архитектуру дискриминатора и <a href="https://github.com/photosynthesis-team/piq">piq</a> за Content-лосс.

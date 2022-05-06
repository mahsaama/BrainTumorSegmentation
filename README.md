# BrainTumorSegmentation

Dataset
-----------------------------------
Dataset is available in [here](https://ipp.cbica.upenn.edu/categories/brats2020).

1. Training: 3D MRI images from `369` patients (Height=240, Width=240, Depth=155)
    - Images: 4 modalities 
    - Masks: 4 classes(0, 1, 2, and 4)

![training image](images/training_image.png) 

2. Validation: 3D MRI images from `125` patients (Height=240, Width=240, Depth=155)
    - Images: 4 modalities

![validation image](images/validation_image.png)


Preprocess and Augmentation
---------------------------------------
1. Flip
![horizontal flip](images/flip_x.png)
![vertical flip](images/flip_y.png)

2. Brightness
![brightness](images/brightness.png)

3. Rotation
![rotation](images/rotation.png)

4. Elastic
![elastic](images/elastic.png)

5. Shift
![shift](images/shift.png)

6. Swirl
![swirl](images/swirl.png)

Models
-----------------------------------
1. UNet3D
![unet3d](images/unet.png)

2. AttUnet3D
![attunet3d](images/attention_unet.png)

3. GAN (Vox2Vox)
![gan generator](images/gan_generator.webp)
![gan discriminator](images/gan_discriminator.webp)

Metrics
------------------------------------
1. Dice Loss

2. Generator Loss

3. Discriminator Loss

Requirements
-----------------------------------
- tensorflow
- tensorflow-addons
- tensorflow-cpu
- numpy
- matplotlib
- elasticdeform
- scikit-learn
- scipy
- nibabel
- SimpleITK

Install the requirements using command below:
```bash
pip install -r requirements.txt 
```
Run
--------------------------------------

Use the following command to see the arguments needed for running:
```bash
python -m scripts.main -h
```

For running **UNet3D** model use this:
```bash
python -m scripts.main -bs 4 -ps 128 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -np 2 -aug 1 -m unet
```

For running **AttUnet3D** model use this:
```bash
python -m scripts.main -bs 4 -ps 128 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -np 2 -aug 1 -m att_unet
```

For running **GAN** model use this:
```bash
python -m scripts.main -bs 4 -ps 128 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -np 2 -aug 1 -m gan
```

References
---------------------------------------------------
- [Vox2Vox: 3D-GAN for Brain Tumour Segmentation](https://github.com/mdciri/Vox2Vox)
- [Attention Unet](https://github.com/sfczekalski/attention_unet)
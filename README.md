# BrainTumorSegmentation

Requirements:
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
```pip install -r requirements.txt ```

Use the following command to see the arguments needed for running:
```python -m scripts.main -h```

For running **UNet3D** model use this:
```python -m scripts.main -nc 4 -bs 4 -ps 32 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -m unet```

For running **AttUnet3D** model use this:
```python -m scripts.main -nc 4 -bs 4 -ps 32 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -m att_unet```

For running **GAN** model use this:
```python -m scripts.main -nc 4 -bs 4 -ps 32 -a 5 -ne 1 -ef 0.25 -lr 1e-3 -b1 0.9 -ds 100 -m gan```
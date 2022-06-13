import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from .augmentation import aug_batch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os


def show_folder_images(url):
    fig = plt.figure(figsize=(15, 3))
    i = 0
    for root, dirs, files in os.walk(url):
        for fil in files:
            i += 1
            path = os.path.join(url, fil)
            image = sitk.ReadImage(path)
            z = int(image.GetDepth() / 2)
            img = sitk.GetArrayViewFromImage(image)[z, :, :]
            fig.add_subplot(1, len(files), i)
            plt.imshow(img)
            plt.title(fil.split("_")[-1].split(".")[0])


def sec_to_minute(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)) % 24)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def load_img(img_files):
    """Load one image and its target form file"""
    N = len(img_files)
    # target
    y = nib.load(img_files[N - 1]).get_fdata(dtype="float32", caching="unchanged")
    y = y[40:200, 34:226, 8:136]
    y[y == 4] = 3

    X_norm = np.empty((240, 240, 155, 4))
    for channel in range(N - 1):
        X = nib.load(img_files[channel]).get_fdata(dtype="float32", caching="unchanged")
        brain = X[X != 0]
        brain_norm = np.zeros_like(X)  # background at -100
        norm = (brain - np.mean(brain)) / np.std(brain)
        brain_norm[X != 0] = norm
        X_norm[:, :, :, channel] = brain_norm

    X_norm = X_norm[40:200, 34:226, 8:136, :]
    del (X, brain, brain_norm)

    return X_norm, y

def patch_extraction(Xb, yb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """

    batch_size, rows, columns, slices, channels = Xb.shape
    Npatches = (rows * columns * slices) // (sizePatches ** 3)
    X_patches = np.empty(
        (batch_size * Npatches, sizePatches, sizePatches, sizePatches, channels)
    )
    y_patches = np.empty((batch_size * Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for m in range(0, rows, sizePatches):
            for n in range(0, columns, sizePatches):
                for o in range(0, slices, sizePatches):
                    X_patches[i] = Xb[
                        b, m : m + sizePatches, n : n + sizePatches, o : o + sizePatches, :
                    ]
                    y_patches[i] = yb[
                        b, m : m + sizePatches, n : n + sizePatches, o : o + sizePatches
                    ]
                    i += 1
        # for p in range(Npatches):
        #     x = np.random.randint(rows - sizePatches + 1)
        #     y = np.random.randint(columns - sizePatches + 1)
        #     z = np.random.randint(slices - sizePatches + 1)

        #     X_patches[i] = Xb[
        #         b, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches, :
        #     ]
        #     y_patches[i] = yb[
        #         b, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches
        #     ]
        #     i += 1

    return X_patches, y_patches

class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        batch_size=4,
        dim=(160, 192, 128),
        n_channels=4,
        n_classes=4,
        shuffle=True,
        augmentation=False,
        patch_size=128,
        n_patches=1,
    ):
        "Initialization"
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)

        if index == self.__len__() - 1:
            self.on_epoch_end()

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)

        X_aug, y_aug = patch_extraction(
            X, y, sizePatches=self.patch_size, Npatches=self.n_patches
        )
        
        if self.augmentation == True:
            return X_aug.astype("float32"), y_aug
        else:
            return X_aug.astype("float32"), to_categorical(y_aug, self.n_classes)

    def __data_augmentation(self, X, y):
        "Apply augmentation"
        X_aug, y_aug = aug_batch(X, y)
        return X_aug, to_categorical(y_aug, self.n_classes)

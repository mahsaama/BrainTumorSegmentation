import argparse
import numpy as np
import glob
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.utils import DataGenerator
from models.unet_model import UNet3D
from models.gan_model import GAN
from models.att_unet_model import AttUnet3D
from models.att_gan_model import AttGAN
from models.unet_dcn_model import UNet3D_DCN
from models.att_unet_dcn_model import AttUnet3DDCN
from models.att_gan_dcn_model import AttGANDCN

# define seeds to genetare predictable results
random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)

parser = argparse.ArgumentParser("BTS Training and validation", add_help=False)

# training parameters
parser.add_argument("-bs", "--batch_size", default=4, type=int, help="batch size")
parser.add_argument("-ps", "--patch_size", default=128, type=int, help="patch size")
parser.add_argument("-a", "--alpha", default=5, type=int, help="alpha weight")
parser.add_argument("-ne", "--num_epochs", default=1, type=int, help="number of epochs")
parser.add_argument(
    "-ef",
    "--eval_frac",
    default=0.25,
    type=float,
    help="fraction of data for evaluation",
)
parser.add_argument("-m", "--model", default="unet", type=str, help="model")
parser.add_argument(
    "-lr", "--learning_rate", default=1e-3, type=float, help="learning rate"
)
parser.add_argument(
    "-b1", "--beta_1", default=0.9, type=float, help="beta1 for momentum"
)
parser.add_argument(
    "-ds", "--dataset", default=2020, type=int, help="Dataset to use"
)
parser.add_argument("-np", "--num_patch", default=1, type=int, help="number of patches")
parser.add_argument(
    "-aug",
    "--augmentation",
    default=1,
    type=int,
    help="whether augment the data or not",
)

args = parser.parse_args()
n_classes = 4
batch_size = args.batch_size
patch_size = args.patch_size
alpha = args.alpha
n_epochs = args.num_epochs
eval_frac = args.eval_frac
model = args.model
lr = args.learning_rate
beta_1 = args.beta_1
ds = args.dataset
num_patches = args.num_patch
aug = True if args.augmentation == 1 else False

classes = np.arange(n_classes)

# class weights
class_weights = np.array([0.25659472, 45.465614, 16.543337, 49.11155], dtype="f")

if ds == 2020:
    p = "../Dataset_BRATS_2020/Training/"
elif ds == 2021:
    p = "../Dataset_BRATS_2021/"
    
# images lists
t1_list = sorted(glob.glob(p+"*/*t1.nii.gz"))[:10]
t2_list = sorted(glob.glob(p+"*/*t2.nii.gz"))[:10]
t1ce_list = sorted(glob.glob(p+"*/*t1ce.nii.gz"))[:10]
flair_list = sorted(glob.glob(p+"*/*flair.nii.gz"))[:10]
seg_list = sorted(glob.glob(p+"*/*seg.nii.gz"))[:10]


# create the training and validation sets
n_data = len(t1_list)
idx = np.arange(n_data)

idxTrain, idxValid = train_test_split(idx, test_size=eval_frac)
sets = {"train": [], "valid": []}

for i in idxTrain:
    sets["train"].append(
        [t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]]
    )
for i in idxValid:
    sets["valid"].append(
        [t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]]
    )


train_gen = DataGenerator(
    sets["train"],
    batch_size=batch_size,
    n_classes=n_classes,
    augmentation=aug,
    patch_size=patch_size,
    n_patches=num_patches,
)
valid_gen = DataGenerator(
    sets["valid"],
    batch_size=batch_size,
    n_classes=n_classes,
    augmentation=aug,
    patch_size=patch_size,
    n_patches=num_patches,
)

results_path = os.path.join(".", "RESULTS")
if os.path.exists(results_path) == False:
    os.mkdir(results_path)


if model == "unet":
    # train the unet model
    path = os.path.join(".", "RESULTS", model)
    unet = UNet3D(patch_size, n_classes, class_weights, path, lr, beta_1)
    history = unet.train(train_gen, valid_gen, n_epochs)

elif model == "att_unet":
    # train the attention unet model
    path = os.path.join(".", "RESULTS", model)
    att_unet = AttUnet3D(patch_size, n_classes, class_weights, path, lr, beta_1)
    history = att_unet.train(train_gen, valid_gen, n_epochs)

elif model == "gan":
    # train the vox2vox model
    path = os.path.join(".", "RESULTS", model)
    gan = GAN(patch_size, n_classes, class_weights, path, lr, beta_1, alpha)
    history = gan.train(train_gen, valid_gen, n_epochs)

elif model == "att_gan":
    # train the vox2vox model with attention in generator
    path = os.path.join(".", "RESULTS", model)
    gan = AttGAN(patch_size, n_classes, class_weights, path, lr, beta_1, alpha)
    history = gan.train(train_gen, valid_gen, n_epochs)

if model == "unet_dc":
    # train the unet model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    unet_dc = UNet3D_DCN(
        batch_size*num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    history = unet_dc.train(train_gen, valid_gen, n_epochs)

if model == "att_unet_dc":
    # train the attention unet model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    att_unet_dc = AttUnet3DDCN(
        batch_size*num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    history = att_unet_dc.train(train_gen, valid_gen, n_epochs)
    
if model == "att_gan_dc":
    # train the attention gan model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    att_gan_dc = AttGANDCN(
        batch_size*num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    history = att_gan_dc.train(train_gen, valid_gen, n_epochs)
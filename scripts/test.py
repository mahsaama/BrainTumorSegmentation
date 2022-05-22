import argparse
import numpy as np
import glob
import os
import random
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


parser = argparse.ArgumentParser("BTS Training and validation", add_help=False)

# training parameters
parser.add_argument("-bs", "--batch_size", default=4, type=int, help="batch size")
parser.add_argument("-ps", "--patch_size", default=128, type=int, help="patch size")
parser.add_argument("-a", "--alpha", default=5, type=int, help="alpha weight")
parser.add_argument(
    "-nt", "--num_tests", default=1, type=int, help="number of test images"
)
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
parser.add_argument("-ds", "--dataset", default=2020, type=int, help="Dataset to use")
parser.add_argument("-np", "--num_patch", default=1, type=int, help="number of patches")
parser.add_argument(
    "-aug",
    "--augmentation",
    default=0,
    type=int,
    help="whether augment the data or not",
)

args = parser.parse_args()
n_classes = 4
batch_size = args.batch_size
patch_size = args.patch_size
alpha = args.alpha
n_tests = args.num_tests
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
t1_list = sorted(glob.glob(p + "*/*t1.nii.gz"))[:n_tests]
t2_list = sorted(glob.glob(p + "*/*t2.nii.gz"))[:n_tests]
t1ce_list = sorted(glob.glob(p + "*/*t1ce.nii.gz"))[:n_tests]
flair_list = sorted(glob.glob(p + "*/*flair.nii.gz"))[:n_tests]
seg_list = sorted(glob.glob(p + "*/*seg.nii.gz"))[:n_tests]


# create the training and validation sets
n_data = len(t1_list)
test_data = []

for i in range(n_data):
    test_data.append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])


test_gen = DataGenerator(
    test_data,
    batch_size=batch_size,
    n_classes=n_classes,
    augmentation=aug,
    patch_size=patch_size,
    n_patches=num_patches,
)


results_path = os.path.join(".", "RESULTS")


if model == "unet":
    # train the unet model
    path = os.path.join(".", "RESULTS", model)
    unet = UNet3D(patch_size, n_classes, class_weights, path, lr, beta_1)
    unet.load_weights(path + "/" + model + "/UNET.h5")
    unet.predict(test_gen)

elif model == "att_unet":
    # train the attention unet model
    path = os.path.join(".", "RESULTS", model)
    att_unet = AttUnet3D(patch_size, n_classes, class_weights, path, lr, beta_1)
    att_unet.load_weights(path + "/" + model + "/Att_UNET.h5")
    att_unet.predict(test_gen)

elif model == "gan":
    # train the vox2vox model
    path = os.path.join(".", "RESULTS", model)
    gan = GAN(patch_size, n_classes, class_weights, path, lr, beta_1, alpha)
    gan.G.load_weights(path + "/" + model + "/Generator.h5")
    gan.D.load_weights(path + "/" + model + "/Discriminator.h5")
    gan.predict(test_gen)

elif model == "att_gan":
    # train the vox2vox model with attention in generator
    path = os.path.join(".", "RESULTS", model)
    att_gan = AttGAN(patch_size, n_classes, class_weights, path, lr, beta_1, alpha)
    att_gan.G.load_weights(path + "/" + model + "/Generator.h5")
    att_gan.D.load_weights(path + "/" + model + "/Discriminator.h5")
    att_gan.predict(test_gen)
    
if model == "unet_dc":
    # train the unet model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    unet_dc = UNet3D_DCN(
        batch_size * num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    unet_dc.load_weights(path + "/" + model + "/UNET_DCN.h5")
    unet_dc.predict(test_gen)

if model == "att_unet_dc":
    # train the attention unet model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    att_unet_dc = AttUnet3DDCN(
        batch_size * num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    att_unet_dc.load_weights(path + "/" + model + "/att_UNET_DCN.h5")
    att_unet_dc.predict(test_gen)

if model == "att_gan_dc":
    # train the attention gan model with deformable convolution
    path = os.path.join(".", "RESULTS", model)
    att_gan_dc = AttGANDCN(
        batch_size * num_patches, patch_size, n_classes, class_weights, path, lr, beta_1
    )
    os.chdir(path + "/" + model + "/")
    att_gan_dc.G.load_weights(path + "/" + model + "/Generator.h5")
    att_gan_dc.D.load_weights(path + "/" + model + "/Discriminator.h5")
    os.chdir("../../")
    att_gan_dc.predict(test_gen)

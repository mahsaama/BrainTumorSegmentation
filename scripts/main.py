import argparse
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

from utils.utils import DataGenerator
from models.unet_model import UNet3D
from models.gan_model import GAN
from models.att_unet_model import AttUnet3D


parser = argparse.ArgumentParser("BTS Training and validation", add_help=False)

# training parameters
parser.add_argument(
    "-nc", "--num_classes", default=4, type=int, help="number of classes"
)
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


args = parser.parse_args()
n_classes = args.num_classes
batch_size = args.batch_size
patch_size = args.patch_size
alpha = args.alpha
n_epochs = args.num_epochs
eval_frac = args.eval_frac
model = args.model
lr = args.learning_rate
beta_1 = args.beta_1

classes = np.arange(n_classes)
# class weights
class_weights = np.array([0.25659472, 45.465614, 16.543337, 49.11155], dtype="f")

# uncomment the code below to see the input images modalities and mask
# show_folder_images("../Dataset_BRATS_2020/Training/BraTS20_Training_001/")
# show_folder_images("../Dataset_BRATS_2020/Validation/BraTS20_Validation_001/")

# images lists
t1_list = sorted(glob.glob("../Dataset_BRATS_2020/Training/*/*t1.nii.gz"))
t2_list = sorted(glob.glob("../Dataset_BRATS_2020/Training/*/*t2.nii.gz"))
t1ce_list = sorted(glob.glob("../Dataset_BRATS_2020/Training/*/*t1ce.nii.gz"))
flair_list = sorted(glob.glob("../Dataset_BRATS_2020/Training/*/*flair.nii.gz"))
seg_list = sorted(glob.glob("../Dataset_BRATS_2020/Training/*/*seg.nii.gz"))

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
    augmentation=True,
    patch_size=patch_size,
)
valid_gen = DataGenerator(
    sets["valid"],
    batch_size=batch_size,
    n_classes=n_classes,
    augmentation=True,
    patch_size=patch_size,
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
    path = os.path.join(".", "RESULTS", model)
    att_unet = AttUnet3D()
    history = att_unet.train(train_gen, valid_gen, n_epochs)

elif model == "gan":
    # train the vox2vox model
    path = os.path.join(".", "RESULTS", model)
    gan = GAN()
    history = gan.train(train_gen, valid_gen, alpha, n_epochs)

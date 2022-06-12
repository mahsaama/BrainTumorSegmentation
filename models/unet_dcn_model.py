import os
import numpy as np
import tensorflow as tf
from utils.losses import diceLoss, per_class_dice
from utils.utils import sec_to_minute
import matplotlib.image as mpim
from sys import stdout
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dropout,
    concatenate,
    MaxPooling3D,
    UpSampling3D,
    Activation,
    Conv3D,
)
from utils.deformable_conv_3d import DCN3D
from sklearn.metrics import confusion_matrix


class UNet3D_DCN:
    def __init__(
        self,
        batch_size,
        patch_size,
        n_classes,
        class_weights,
        path,
        lr=2e-4,
        beta_1=0.5,
    ):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        self.path = path
        self.model = self.build()

    def build(self):
        inputs = Input(
            (self.patch_size, self.patch_size, self.patch_size, self.n_classes),
            name="input_image",
        )
        # conv1 = Conv3D(
        #     64, 3, activation="relu", padding="same", data_format="channels_last"
        # )(inputs)
        conv1 = DCN3D(64, 3, self.batch_size, activation="relu")(inputs)
        # conv1 = Conv3D(64, 3, activation="relu", padding="same")(conv1)
        conv1 = DCN3D(64, 3, self.batch_size, activation="relu")(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        # conv2 = Conv3D(128, 3, activation="relu", padding="same")(pool1)
        conv2 = DCN3D(128, 3, self.batch_size, activation="relu")(pool1)
        # conv2 = Conv3D(128, 3, activation="relu", padding="same")(conv2)
        conv2 = DCN3D(128, 3, self.batch_size, activation="relu")(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        # conv3 = Conv3D(256, 3, activation="relu", padding="same")(pool2)
        conv3 = DCN3D(256, 3, self.batch_size, activation="relu")(pool2)
        # conv3 = Conv3D(256, 3, activation="relu", padding="same")(conv3)
        conv3 = DCN3D(256, 3, self.batch_size, activation="relu")(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        # conv4 = Conv3D(512, 3, activation="relu", padding="same")(pool3)
        conv4 = DCN3D(512, 3, self.batch_size, activation="relu")(pool3)
        # conv4 = Conv3D(512, 3, activation="relu", padding="same")(conv4)
        conv4 = DCN3D(512, 3, self.batch_size, activation="relu")(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

        # conv5 = Conv3D(1024, 3, activation="relu", padding="same")(pool4)
        conv5 = DCN3D(1024, 3, self.batch_size, activation="relu")(pool4)
        # conv5 = Conv3D(1024, 3, activation="relu", padding="same")(conv5)
        conv5 = DCN3D(1024, 3, self.batch_size, activation="relu")(conv5)
        drop5 = Dropout(0.5)(conv5)

        # up6 = Conv3D(512, 2, activation="relu", padding="same")(
        #     UpSampling3D(size=(2, 2, 2))(drop5)
        # )
        up6 = DCN3D(512, 2, self.batch_size, activation="relu")(
            UpSampling3D(size=(2, 2, 2))(drop5)
        )
        merge6 = concatenate([drop4, up6], axis=-1)
        # conv6 = Conv3D(512, 3, activation="relu", padding="same")(merge6)
        # conv6 = Conv3D(512, 3, activation="relu", padding="same")(conv6)
        conv6 = DCN3D(512, 3, self.batch_size, activation="relu")(merge6)
        conv6 = DCN3D(512, 3, self.batch_size, activation="relu")(conv6)

        # up7 = Conv3D(256, 2, activation="relu", padding="same")(
        #     UpSampling3D(size=(2, 2, 2))(conv6)
        # )
        up7 = DCN3D(256, 2, self.batch_size, activation="relu")(
            UpSampling3D(size=(2, 2, 2))(conv6)
        )
        merge7 = concatenate([conv3, up7], axis=-1)
        # conv7 = Conv3D(256, 3, activation="relu", padding="same")(merge7)
        # conv7 = Conv3D(256, 3, activation="relu", padding="same")(conv7)
        conv7 = DCN3D(256, 3, self.batch_size, activation="relu")(merge7)
        conv7 = DCN3D(256, 3, self.batch_size, activation="relu")(conv7)

        # up8 = Conv3D(128, 2, activation="relu", padding="same")(
        #     UpSampling3D(size=(2, 2, 2))(conv7)
        # )
        up8 = DCN3D(128, 2, self.batch_size, activation="relu")(
            UpSampling3D(size=(2, 2, 2))(conv7)
        )
        merge8 = concatenate([conv2, up8], axis=-1)
        # conv8 = Conv3D(128, 3, activation="relu", padding="same")(merge8)
        # conv8 = Conv3D(128, 3, activation="relu", padding="same")(conv8)
        conv8 = DCN3D(128, 3, self.batch_size, activation="relu")(merge8)
        conv8 = DCN3D(128, 3, self.batch_size, activation="relu")(conv8)

        # up9 = Conv3D(64, 2, activation="relu", padding="same")(
        #     UpSampling3D(size=(2, 2, 2))(conv8)
        # )
        up9 = DCN3D(64, 2, self.batch_size, activation="relu")(
            UpSampling3D(size=(2, 2, 2))(conv8)
        )
        merge9 = concatenate([conv1, up9], axis=-1)
        # conv9 = Conv3D(64, 3, activation="relu", padding="same")(merge9)
        # conv9 = Conv3D(64, 3, activation="relu", padding="same")(conv9)
        # output = Conv3D(4, 1, activation="softmax")(conv9)
        conv9 = DCN3D(64, 3, self.batch_size, activation="relu")(merge9)
        conv9 = DCN3D(64, 3, self.batch_size, activation="relu")(conv9)
        output = Conv3D(4, 1, activation="softmax")(conv9)

        return Model(inputs=inputs, outputs=output, name="UnetDCN")

    @tf.function
    def train_step(self, image, target):
        with tf.GradientTape() as tape:
            output = self.model(image, training=True)
            dice_loss = diceLoss(target, output, self.class_weights)
            dice_per_class = per_class_dice(target, output, self.class_weights)

        gradients = tape.gradient(dice_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        dice_percent = (1 - dice_loss) * 100
        return dice_loss, dice_percent, dice_per_class

    @tf.function
    def test_step(self, image, target):
        output = self.model(image, training=False)
        dice_loss = diceLoss(target, output, self.class_weights)
        dice_percent = (1 - dice_loss) * 100
        dice_per_class = per_class_dice(target, output, self.class_weights)

        return dice_loss, dice_percent, dice_per_class

    def train(self, train_gen, valid_gen, epochs):

        if os.path.exists(self.path) == False:
            os.mkdir(self.path)

        Nt = len(train_gen)
        history = {"train": [], "valid": []}
        prev_loss = np.inf

        epoch_dice_loss = tf.keras.metrics.Mean()
        epoch_dice_loss_percent = tf.keras.metrics.Mean()
        epoch_dice_loss_val = tf.keras.metrics.Mean()
        epoch_dice_loss_percent_val = tf.keras.metrics.Mean()

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            start = time.time()
            b = 0
            for Xb, yb in train_gen:
                b += 1
                losses = self.train_step(Xb, yb)
                epoch_dice_loss.update_state(losses[0])
                epoch_dice_loss_percent.update_state(losses[1])

                stdout.write(
                    "\rBatch: {}/{} - dice_loss: {:.4f} - dice_percentage: {:.4f}% - WT: {:.4f} - TC: {:.4f} - ET: {:.4f} ".format(
                        b,
                        Nt,
                        epoch_dice_loss.result(),
                        epoch_dice_loss_percent.result(),
                        losses[-1][0],
                        losses[-1][1],
                        losses[-1][2],
                    )
                )
                stdout.flush()
            history["train"].append(
                [epoch_dice_loss.result(), epoch_dice_loss_percent.result()]
            )

            for Xb, yb in valid_gen:
                losses_val = self.test_step(Xb, yb)
                epoch_dice_loss_val.update_state(losses_val[0])
                epoch_dice_loss_percent_val.update_state(losses_val[1])

            stdout.write(
                "\n               dice_loss_val: {:.4f} - dice_percentage_val: {:.4f}% - WT: {:.4f} - TC: {:.4f} - ET: {:.4f} ".format(
                    epoch_dice_loss_val.result(),
                    epoch_dice_loss_percent_val.result(),
                    losses[-1][0],
                    losses[-1][1],
                    losses[-1][2],
                )
            )
            stdout.flush()
            history["valid"].append(
                [epoch_dice_loss_val.result(), epoch_dice_loss_percent_val.result()]
            )

            # save pred image at epoch e
            y_pred = self.model.predict(Xb)
            y_true = np.argmax(yb, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)

            print()
            print(
                confusion_matrix(
                    y_true.flatten(),
                    y_pred.flatten(),
                )
            )

            patch_size = valid_gen.patch_size
            canvas = np.zeros((patch_size, patch_size * 3))
            idx = np.random.randint(len(Xb))

            x = Xb[idx, :, :, patch_size // 2, 2]
            canvas[0:patch_size, 0:patch_size] = (x - np.min(x)) / (
                np.max(x) - np.min(x) + 1e-6
            )
            canvas[0:patch_size, patch_size : 2 * patch_size] = (
                y_true[idx, :, :, patch_size // 2] / 3
            )
            canvas[0:patch_size, 2 * patch_size : 3 * patch_size] = (
                y_pred[idx, :, :, patch_size // 2] / 3
            )

            fname = (self.path + "/pred@epoch_{:03d}.png").format(e + 1)
            mpim.imsave(fname, canvas, cmap="gray")

            # save models
            print(" ")
            if epoch_dice_loss_val.result() < prev_loss:
                self.model.save_weights(self.path + "/UNET_DCN.h5")
                print(
                    "Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(
                        prev_loss, epoch_dice_loss_val.result()
                    )
                )
                prev_loss = epoch_dice_loss_val.result()
            else:
                print("Validation loss did not decrese from {:.4f}.".format(prev_loss))

            # resets losses states
            epoch_dice_loss.reset_states()
            epoch_dice_loss_percent.reset_states()
            epoch_dice_loss_val.reset_states()
            epoch_dice_loss_percent_val.reset_states()

            del Xb, yb, canvas, y_pred, y_true, idx
            print("Time: {}\n".format(sec_to_minute(time.time() - start)))
        return history

    def predict(self, test_gen):
        start = time.time()
        i = 0
        for Xb, yb in test_gen:
            i += 1
            dice_loss, dice_acc = self.test_step(Xb, yb)
            # save pred image at epoch e
            y_pred = self.model.predict(Xb)
            y_true = np.argmax(yb, axis=-1)
            y_pred = np.argmax(y_pred, axis=-1)

            patch_size = test_gen.patch_size
            canvas = np.zeros((patch_size, patch_size * 3))
            idx = np.random.randint(len(Xb))

            x = Xb[idx, :, :, patch_size // 2, 2]
            canvas[0:patch_size, 0:patch_size] = (x - np.min(x)) / (
                np.max(x) - np.min(x) + 1e-6
            )
            canvas[0:patch_size, patch_size : 2 * patch_size] = (
                y_true[idx, :, :, patch_size // 2] / 3
            )
            canvas[0:patch_size, 2 * patch_size : 3 * patch_size] = (
                y_pred[idx, :, :, patch_size // 2] / 3
            )

            fname = (self.path + "/test_{:03d}.png").format(i)
            mpim.imsave(fname, canvas, cmap="gray")

            del Xb, yb, canvas, y_pred, y_true, idx
            print("Dice Accuracy: {}".format(dice_acc))
        print("Time: {}".format(sec_to_minute(time.time() - start)))

import os
import numpy as np
import tensorflow as tf
from utils.losses import discriminator_loss, generator_loss, per_class_dice
from utils.utils import sec_to_minute
from sys import stdout
import matplotlib.image as mpim
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Conv3D,
    Conv3DTranspose,
    Dropout,
    LeakyReLU,
    ZeroPadding3D,
    BatchNormalization,
    Add,
    Activation,
    Multiply,
)
from tensorflow_addons.layers import InstanceNormalization
from utils.deformable_conv_3d import DCN3D
from sklearn.metrics import confusion_matrix


class AttGANDCN:
    def __init__(
        self,
        batch_size,
        patch_size,
        n_classes,
        class_weights,
        path,
        lr=2e-4,
        beta_1=0.5,
        alpha=5,
    ):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        self.path = path
        self.alpha = alpha
        self.G, self.D = self.build()

    def Generator(self):
        """
        Generator model
        """

        def attention_block(F_g, F_l, F_int):
            g = Conv3D(F_int, 1, padding="valid")(F_g)
            g = BatchNormalization()(g)

            x = Conv3D(F_int, 1, padding="valid")(F_l)
            x = BatchNormalization()(x)

            psi = Add()([g, x])
            psi = Activation("relu")(psi)

            psi = Conv3D(1, 1, padding="valid")(psi)
            psi = BatchNormalization()(psi)
            psi = Activation("sigmoid")(psi)

            return Multiply()([F_l, psi])

        def encoder_step(layer, Nf, ks, norm=True):
            x = DCN3D(
                Nf, ks, self.batch_size, activation="", kernel_initializer="he_normal"
            )(layer)
            # x = Conv3D(
            #     Nf,
            #     kernel_size=ks,
            #     strides=2,
            #     kernel_initializer="he_normal",
            #     padding="same",
            # )(layer)
            if norm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.2)(x)

            return x

        def bottlenek(layer, Nf, ks):
            x = DCN3D(
                Nf, ks, self.batch_size, activation="", kernel_initializer="he_normal"
            )(layer)
            # x = Conv3D(
            #     Nf,
            #     kernel_size=ks,
            #     strides=2,
            #     kernel_initializer="he_normal",
            #     padding="same",
            # )(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            for i in range(4):
                y = DCN3D(
                    Nf,
                    ks,
                    self.batch_size,
                    activation="",
                    kernel_initializer="he_normal",
                )(x)
                # y = Conv3D(
                #     Nf,
                #     kernel_size=ks,
                #     strides=1,
                #     kernel_initializer="he_normal",
                #     padding="same",
                # )(x)
                x = InstanceNormalization()(y)
                x = LeakyReLU()(x)
                x = Concatenate()([x, y])

            return x

        def decoder_step(layer, layer_to_concatenate, Nf, ks):
            x = Conv3DTranspose(
                Nf,
                kernel_size=ks,
                strides=1,
                # strides=2,
                padding="same",
                kernel_initializer="he_normal",
            )(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            att = attention_block(x, layer_to_concatenate, Nf)
            x = Concatenate()([x, att])
            x = Dropout(0.2)(x)
            return x

        layers_to_concatenate = []
        inputs = Input(
            (self.patch_size, self.patch_size, self.patch_size, self.n_classes),
            name="input_image",
        )
        Nfilter_start = self.patch_size // 2
        ks = 4

        # encoder
        e1 = encoder_step(inputs, Nfilter_start, ks, False)
        layers_to_concatenate.append(e1)
        e2 = encoder_step(e1, Nfilter_start * 2, ks)
        layers_to_concatenate.append(e2)
        e3 = encoder_step(e2, Nfilter_start * 4, ks)
        layers_to_concatenate.append(e3)

        # bottlenek
        b = bottlenek(e3, Nfilter_start * 8, ks)

        # decoder
        d1 = decoder_step(b, layers_to_concatenate.pop(), Nfilter_start * 4, ks)
        d2 = decoder_step(d1, layers_to_concatenate.pop(), Nfilter_start * 2, ks)
        d3 = decoder_step(d2, layers_to_concatenate.pop(), Nfilter_start, ks)

        # classifier
        outputs = Conv3DTranspose(
            4,
            kernel_size=ks,
            strides=1,
            # strides=2,
            padding="same",
            kernel_initializer="he_normal",
            activation="softmax",
            name="output_generator",
        )(d3)

        return Model(inputs=inputs, outputs=outputs, name="Generator")

    def Discriminator(self):
        """
        Discriminator model
        """
        inputs = Input(
            (self.patch_size, self.patch_size, self.patch_size, self.n_classes),
            name="input_image",
        )
        targets = Input(
            (self.patch_size, self.patch_size, self.patch_size, self.n_classes),
            name="target_image",
        )
        Nfilter_start = self.patch_size // 2
        depth = 3
        ks = 4

        def encoder_step(layer, Nf, norm=True):
            x = Conv3D(
                Nf,
                kernel_size=ks,
                strides=1,
                # strides=2,
                kernel_initializer="he_normal",
                padding="same",
            )(layer)
            if norm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.2)(x)

            return x

        x = Concatenate()([inputs, targets])

        for d in range(depth):
            if d == 0:
                x = encoder_step(x, Nfilter_start * np.power(2, d), False)
            else:
                x = encoder_step(x, Nfilter_start * np.power(2, d))

        x = ZeroPadding3D()(x)
        x = Conv3D(
            Nfilter_start * (2 ** depth),
            ks,
            strides=1,
            padding="valid",
            kernel_initializer="he_normal",
        )(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)

        x = ZeroPadding3D()(x)
        last = Conv3D(
            1,
            ks,
            strides=1,
            padding="valid",
            kernel_initializer="he_normal",
            name="output_discriminator",
        )(x)

        return Model(inputs=[targets, inputs], outputs=last, name="Discriminator")

    def build(self):
        return self.Generator(), self.Discriminator()

    @tf.function
    def train_step(self, image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            gen_output = self.G(image, training=True)

            disc_real_output = self.D([image, target], training=True)
            disc_fake_output = self.D([image, gen_output], training=True)

            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            gen_loss, dice_loss, disc_loss_gen, dice_percent = generator_loss(
                target, gen_output, disc_fake_output, self.class_weights, self.alpha
            )

            dice_per_class = per_class_dice(target, gen_output, self.class_weights)

        generator_gradients = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.D.trainable_variables
        )

        self.G_optimizer.apply_gradients(
            zip(generator_gradients, self.G.trainable_variables)
        )
        self.D_optimizer.apply_gradients(
            zip(discriminator_gradients, self.D.trainable_variables)
        )

        return gen_loss, dice_loss, disc_loss_gen, dice_percent, dice_per_class

    @tf.function
    def test_step(self, image, target):
        gen_output = self.G(image, training=False)

        disc_real_output = self.D([image, target], training=False)
        disc_fake_output = self.D([image, gen_output], training=False)

        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        gen_loss, dice_loss, disc_loss_gen, dice_percent = generator_loss(
            target, gen_output, disc_fake_output, self.class_weights, self.alpha
        )

        dice_per_class = per_class_dice(target, gen_output, self.class_weights)

        return gen_loss, dice_loss, disc_loss_gen, dice_percent, dice_per_class

    def train(self, train_gen, valid_gen, epochs):

        if os.path.exists(self.path) == False:
            os.mkdir(self.path)

        Nt = len(train_gen)
        history = {"train": [], "valid": []}
        prev_loss = np.inf

        epoch_v2v_loss = tf.keras.metrics.Mean()
        epoch_dice_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        epoch_dp = tf.keras.metrics.Mean()
        epoch_v2v_loss_val = tf.keras.metrics.Mean()
        epoch_dice_loss_val = tf.keras.metrics.Mean()
        epoch_disc_loss_val = tf.keras.metrics.Mean()
        epoch_dp_val = tf.keras.metrics.Mean()

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            start = time.time()
            b = 0
            for Xb, yb in train_gen:
                b += 1
                losses = self.train_step(Xb, yb)
                epoch_v2v_loss.update_state(losses[0])
                epoch_dice_loss.update_state(losses[1])
                epoch_disc_loss.update_state(losses[2])
                epoch_dp.update_state(losses[3])

                stdout.write(
                    "\rBatch: {}/{} - loss: {:.4f} - dice_loss: {:.4f} - disc_loss: {:.4f} - dice_percentage: {:.4f}% - WT: {:.4f} - TC: {:.4f} - ET: {:.4f} ".format(
                        b,
                        Nt,
                        epoch_v2v_loss.result(),
                        epoch_dice_loss.result(),
                        epoch_disc_loss.result(),
                        epoch_dp.result(),
                        losses[-1][0],
                        losses[-1][1],
                        losses[-1][2],
                    )
                )
                stdout.flush()
            history["train"].append(
                [
                    epoch_v2v_loss.result(),
                    epoch_dice_loss.result(),
                    epoch_disc_loss.result(),
                    epoch_dp.result(),
                ]
            )

            for Xb, yb in valid_gen:
                losses_val = self.test_step(Xb, yb)
                epoch_v2v_loss_val.update_state(losses_val[0])
                epoch_dice_loss_val.update_state(losses_val[1])
                epoch_disc_loss_val.update_state(losses_val[2])
                epoch_dp_val.update_state(losses_val[3])

            stdout.write(
                "\n               loss_val: {:.4f} - dice_loss_val: {:.4f} - disc_loss_val: {:.4f} - dice_percentage_val: {:.4f}% - WT: {:.4f} - TC: {:.4f} - ET: {:.4f} ".format(
                    epoch_v2v_loss_val.result(),
                    epoch_dice_loss_val.result(),
                    epoch_disc_loss_val.result(),
                    epoch_dp_val.result(),
                    losses_val[-1][0],
                    losses_val[-1][1],
                    losses_val[-1][2],
                )
            )
            stdout.flush()
            history["valid"].append(
                [
                    epoch_v2v_loss_val.result(),
                    epoch_dice_loss_val.result(),
                    epoch_disc_loss_val.result(),
                    epoch_dp_val.result(),
                ]
            )

            # save pred image at epoch e
            y_pred = self.G.predict(Xb)
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
            if epoch_v2v_loss_val.result() < prev_loss:
                self.G.save_weights(self.path + "/Generator.h5")
                self.D.save_weights(self.path + "/Discriminator.h5")
                print(
                    "Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(
                        prev_loss, epoch_v2v_loss_val.result()
                    )
                )
                prev_loss = epoch_v2v_loss_val.result()
            else:
                print("Validation loss did not decrese from {:.4f}.".format(prev_loss))

            # resets losses states
            epoch_v2v_loss.reset_states()
            epoch_dice_loss.reset_states()
            epoch_disc_loss.reset_states()
            epoch_dp.reset_states()
            epoch_v2v_loss_val.reset_states()
            epoch_dice_loss_val.reset_states()
            epoch_disc_loss_val.reset_states()
            epoch_dp_val.reset_states()

            del Xb, yb, canvas, y_pred, y_true, idx
            print("Time: {}\n".format(sec_to_minute(time.time() - start)))
        return history

    def predict(self, test_gen):
        start = time.time()
        i = 0
        for Xb, yb in test_gen:
            i += 1
            gen_loss, dice_loss, disc_loss_gen, dice_acc = self.test_step(Xb, yb)
            # save pred image at epoch e
            y_pred = self.G.predict(Xb)
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

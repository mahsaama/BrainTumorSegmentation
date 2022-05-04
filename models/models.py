import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Concatenate, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, concatenate, ZeroPadding3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

def UNet3D(patch_size):
    inputs = Input((patch_size,patch_size,patch_size,4), name='input_image')
    x = inputs

    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last")(x)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3D(256, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3D(128, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(4, 1, activation = 'softmax')(conv9)
   
    return Model(inputs=inputs, outputs=conv10, name='Unet')


def Generator(patch_size):
    '''
    Generator model
    '''
    def encoder_step(layer, Nf, ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(4):
            y = Conv3D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(y)
            x = LeakyReLU()(x)
            x = Concatenate()([x, y])

        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x = Conv3DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = Dropout(0.2)(x)
        return x

    layers_to_concatenate = []
    inputs = Input((patch_size,patch_size,patch_size,4), name='input_image')
    Nfilter_start = patch_size//2
    depth = 4
    ks = 4
    x = inputs

    # encoder
    for d in range(depth-1):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)

    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)

    # classifier
    last = Conv3DTranspose(4, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)
   
    return Model(inputs=inputs, outputs=last, name='Generator')

def Discriminator(patch_size):
    '''
    Discriminator model
    '''
    inputs = Input((patch_size,patch_size,patch_size,4), name='input_image')
    targets = Input((patch_size,patch_size,patch_size,4), name='target_image')
    Nfilter_start = patch_size//2
    depth = 3
    ks = 4

    def encoder_step(layer, Nf, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        return x

    x = Concatenate()([inputs, targets])

    for d in range(depth):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d))
            
    x = ZeroPadding3D()(x)
    x = Conv3D(Nfilter_start*(2**depth), ks, strides=1, padding='valid', kernel_initializer='he_normal')(x) 
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
      
    x = ZeroPadding3D()(x)
    last = Conv3D(1, ks, strides=1, padding='valid', kernel_initializer='he_normal', name='output_discriminator')(x) 

    return Model(inputs=[targets, inputs], outputs=last, name='Discriminator')

def ensembler(imShape, gtShape, class_weights, patch_size):

    start = Input((patch_size,patch_size,patch_size,40))
    fin = Conv3D(4, kernel_size=3, kernel_initializer='he_normal', padding='same', activation='softmax')(start)

    return Model(inputs=start, outputs=fin, name='Ensembler')

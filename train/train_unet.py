import os
import numpy as np
import tensorflow as tf
from models import UNet3D
from losses import diceLoss
import matplotlib.image as mpim
from sys import stdout
import time


# class weights
class_weights = np.array([0.25659472, 45.465614, 16.543337, 49.11155], dtype='f')

# Models
unet = UNet3D(patch_size=64)

# Optimizers
unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(image, target, alpha):
    with tf.GradientTape() as tape:
        output = unet(image, training=True)
        dice_loss = diceLoss(target, output, class_weights)
        
    gradients = tape.gradient(dice_loss, unet.trainable_variables)
    unet_optimizer.apply_gradients(zip(gradients, unet.trainable_variables))
    dice_percent = (1-dice_loss) * 100
    return dice_loss, dice_percent
        
@tf.function
def test_step(image, target, alpha):
    output = unet(image, training=False)
    dice_loss = diceLoss(target, output, class_weights)
    dice_percent = (1-dice_loss) * 100
    return dice_loss, dice_percent

def fit_unet(train_gen, valid_gen, alpha, epochs):
    
    path = './RESULTS_UNET' 
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    Nt = len(train_gen)
    history = {'train': [], 'valid': []}
    prev_loss = np.inf
    
    epoch_dice_loss = tf.keras.metrics.Mean()
    epoch_dice_loss_percent = tf.keras.metrics.Mean()
    epoch_dice_loss_val = tf.keras.metrics.Mean()
    epoch_dice_loss_percent_val = tf.keras.metrics.Mean()
    
    for e in range(epochs):
        print('Epoch {}/{}'.format(e+1,epochs))
        start = time.time()
        b = 0
        for Xb, yb in train_gen:
            b += 1
            losses = train_step(Xb, yb, alpha)
            epoch_dice_loss.update_state(losses[0])
            epoch_dice_loss_percent.update_state(losses[1])
            
            stdout.write('\rBatch: {}/{} - dice_loss: {:.4f} - dice_percentage: {:.4f}% '
                         .format(b, Nt, epoch_dice_loss.result(), epoch_dice_loss_percent.result()))
            stdout.flush()
        history['train'].append([epoch_dice_loss.result(), epoch_dice_loss_percent.result()])
        
        for Xb, yb in valid_gen:
            losses_val = test_step(Xb, yb, alpha)
            epoch_dice_loss_val.update_state(losses_val[0])
            epoch_dice_loss_percent_val.update_state(losses_val[1])
            
        stdout.write('\n               loss_val: {:.4f} - dice_loss_val: {:.4f} - dice_percentage: {:.4f}% '
                     .format(epoch_dice_loss_val.result(), epoch_dice_loss_percent_val.result()))
        stdout.flush()
        history['valid'].append([epoch_dice_loss_val.result(), epoch_dice_loss_percent_val.result()])
        
        # save pred image at epoch e 
        y_pred = unet.predict(Xb)
        y_true = np.argmax(yb, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        patch_size = valid_gen.patch_size
        canvas = np.zeros((patch_size, patch_size*3))
        idx = np.random.randint(len(Xb))
        
        x = Xb[idx,:,:,patch_size//2,2] 
        canvas[0:patch_size, 0:patch_size] = (x - np.min(x))/(np.max(x)-np.min(x)+1e-6)
        canvas[0:patch_size, patch_size:2*patch_size] = y_true[idx,:,:,patch_size//2]/3
        canvas[0:patch_size, 2*patch_size:3*patch_size] = y_pred[idx,:,:,patch_size//2]/3
        
        fname = (path + '/pred@epoch_{:03d}.png').format(e+1)
        mpim.imsave(fname, canvas, cmap='gray')
        
        # save models
        print(' ')
        if epoch_dice_loss_val.result() < prev_loss:    
            unet.save_weights(path + '/UNET.h5')
            print("Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(prev_loss, epoch_dice_loss_val.result()))
            prev_loss = epoch_dice_loss_val.result()
        else:
            print("Validation loss did not decrese from {:.4f}.".format(prev_loss))
        
        # resets losses states
        epoch_dice_loss.reset_states()
        epoch_dice_loss_percent.reset_states()
        epoch_dice_loss_val.reset_states()
        epoch_dice_loss_percent_val.reset_states()
        
        del Xb, yb, canvas, y_pred, y_true, idx
        print('Epoch {}, Time: {}\n'.format(e+1,time.time()-start))        
    return history

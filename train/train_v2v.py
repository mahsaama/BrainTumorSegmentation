import os
import numpy as np
import tensorflow as tf
from models import *
from losses import *
import matplotlib.image as mpim
from sys import stdout
import time


# class weights
class_weights = np.load('class_weights.npy')

# Models
G = Generator(patch_size=128)
D = Discriminator(patch_size=128)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(image, target, alpha):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = G(image, training=True)
        print(gen_output.shape)
        disc_real_output = D([image, target], training=True)
        disc_fake_output = D([image, gen_output], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        
        gen_loss, dice_loss, disc_loss_gen, dice_percent = generator_loss(target, gen_output, disc_fake_output, class_weights, alpha)

    generator_gradients = gen_tape.gradient(gen_loss, G.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D.trainable_variables))
    
    # hd = directed_hausdorff(target.numpy(), gen_output.numpy())[0]
    return gen_loss, dice_loss, disc_loss_gen, dice_percent
    # , hd
        
@tf.function
def test_step(image, target, alpha):
    gen_output = G(image, training=False)

    disc_real_output = D([image, target], training=False)
    disc_fake_output = D([image, gen_output], training=False)
    disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

    gen_loss, dice_loss, disc_loss_gen, dice_percent = generator_loss(target, gen_output, disc_fake_output, class_weights, alpha)
    # hd = directed_hausdorff(target.numpy(), gen_output.numpy())[0]    
    return gen_loss, dice_loss, disc_loss_gen, dice_percent
    # , hd

def fit(train_gen, valid_gen, alpha, epochs):
    
    path = './RESULTS' 
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    Nt = len(train_gen)
    history = {'train': [], 'valid': []}
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
        print('Epoch {}/{}'.format(e+1,epochs))
        start = time.time()
        b = 0
        for Xb, yb in train_gen:
            print(Xb.shape, yb.shape)
            b += 1
            losses = train_step(Xb, yb, alpha)
            epoch_v2v_loss.update_state(losses[0])
            epoch_dice_loss.update_state(losses[1])
            epoch_disc_loss.update_state(losses[2])
            epoch_dp.update_state(losses[3])
            
            stdout.write('\rBatch: {}/{} - loss: {:.4f} - dice_loss: {:.4f} - disc_loss: {:.4f} - dice_percentage: {:.4f}% '
                         .format(b, Nt, epoch_v2v_loss.result(), epoch_dice_loss.result(), epoch_disc_loss.result(), epoch_dp.result()))
            stdout.flush()
        history['train'].append([epoch_v2v_loss.result(), epoch_dice_loss.result(), epoch_disc_loss.result(), epoch_dp.result()])
        
        for Xb, yb in valid_gen:
            losses_val = test_step(Xb, yb, alpha)
            epoch_v2v_loss_val.update_state(losses_val[0])
            epoch_dice_loss_val.update_state(losses_val[1])
            epoch_disc_loss_val.update_state(losses_val[2])
            epoch_dp_val.update_state(losses_val[3])
            
        stdout.write('\n               loss_val: {:.4f} - dice_loss_val: {:.4f} - disc_loss_val: {:.4f} - dice_percentage: {:.4f}% '
                     .format(epoch_v2v_loss_val.result(), epoch_dice_loss_val.result(), epoch_disc_loss_val.result(), epoch_dp_val.result()))
        stdout.flush()
        history['valid'].append([epoch_v2v_loss_val.result(), epoch_dice_loss_val.result(), epoch_disc_loss_val.result(), epoch_dp_val.result()])
        
        # save pred image at epoch e 
        y_pred = G.predict(Xb)
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
        if epoch_v2v_loss_val.result() < prev_loss:    
            G.save_weights(path + '/Generator.h5') 
            D.save_weights(path + '/Discriminator.h5')
            print("Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(prev_loss, epoch_v2v_loss_val.result()))
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
        print('Epoch {}, Time: {}\n'.format(e+1,time.time()-start))        
    return history

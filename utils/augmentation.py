import numpy as np
from scipy.ndimage.interpolation import affine_transform
import elasticdeform
import multiprocessing as mp
from scipy.ndimage import shift
from skimage.transform import swirl
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def patch_extraction(Xb, yb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """
    
    batch_size, rows, columns, slices, channels = Xb.shape
    X_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches, channels))
    y_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for p in range(Npatches):
            x = np.random.randint(rows-sizePatches+1) 
            y = np.random.randint(columns-sizePatches+1)
            z = np.random.randint(slices-sizePatches+1) 

            X_patches[i] = Xb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches, :]
            y_patches[i] = yb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches]
            i += 1
                               
    return X_patches, y_patches

def shift3D(X, y, shift_stds=[20, 20, 20]):      
    """
    Perform random shifts along axes.
    """      
    shifts = [truncnorm.rvs(-1, 1, loc=0, scale=std, size=1) for std in shift_stds]
    
    X_shft = np.empty_like(X)
    for channel in range(X.shape[-1]):
        X_shft[:, :, :, channel] = shift(X[:, :, :, channel], shifts, order=3)
    # X_shft = np.stack(X_shft, axis=-1)

    y_shft = shift(y, shifts, order=0)
    # y_shft = y_shft.reshape(y_shft.shape + (1,))

    return X_shft, y_shft

def swirl3D(X, y, radius=100, strenght_std=1):
    AXES = [(0, 1), (1, 2), (0, 2)]
    ax1, ax2 = random.choice(AXES)
    strenght = truncnorm.rvs(-1, 1, loc=0, scale=strenght_std, size=1)

    # X
    X_swapped = np.swapaxes(X, ax1, ax2)
    for channel in range(X.shape[-1]):
        X_swapped[:, :, :, channel] = swirl(X_swapped[:, :, :, channel], rotation=0, strength=strenght, radius=radius, order=3)
    # X_swapped = np.stack(X_swapped, axis=-1)
    X_swapped = np.swapaxes(X_swapped, ax1, ax2)

    # y
    y_swapped = np.swapaxes(y, ax1, ax2)
    y_swapped = swirl(y_swapped, rotation=0, strength=strenght, radius=radius, order=0)
    # y_swapped = y_swapped.reshape(y_swapped.shape + (1,))
    y_swapped = np.swapaxes(y_swapped, ax1, ax2)
    
    return X_swapped, y_swapped

    
def flip3D(X, y):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0: # flip on x
        X_flip, y_flip = X[::-1, :, :, :], y[::-1, :, :]
    if choice == 1: # flip on y
        X_flip, y_flip = X[:, ::-1, :, :], y[:, ::-1, :]
    if choice == 2: # flip on z
        X_flip, y_flip = X[:, :, ::-1, :], y[:, :, ::-1]
        
    return X_flip, y_flip


def rotation3D(X, y):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.pi*np.random.random_sample(3,)/2
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    R = np.dot(np.dot(Rx, Ry), Rz)
    
    X_rot = np.empty_like(X)
    for channel in range(X.shape[-1]):
        X_rot[:,:,:,channel] = affine_transform(X[:,:,:,channel], R, offset=0, order=3, mode='constant')
    y_rot = affine_transform(y, R, offset=0, order=0, mode='constant')
    
    return X_rot, y_rot

def brightness(X, y):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.
    
    Gain chosen between [0.9 - 1.1]
    Gamma chosen between [0.9 - 1.1]
    
    new_im = gain * im^gamma
    """
    
    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:,:,:,c]        
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        X_new[:,:,:,c] = im_new 
    
    return X_new, y

def elastic(X, y):
    """
    Elastic deformation on a image and its target
    """  
    [Xel, yel] = elasticdeform.deform_random_grid([X, y], sigma=2, axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0], mode='constant')
    
    return Xel, yel

def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """
    
    decisions = np.zeros((N, 4)) # 4 is number of aug techniques to combine (patch extraction excluded)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=4)
        
    return decisions

def combine_aug(X, y, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    
    # make sure to use at least the 25% of original images
    if np.random.random_sample() > 0.75:
        return Xnew, ynew
    
    else:  
        # for i in range(4):
        #     if do[i] == 0:
        #         Xnew, ynew = flip3D(Xnew, ynew)
        #     elif do[i] == 1:
        #         Xnew, ynew = brightness(Xnew, ynew)   
        #     elif do[i] == 2:
        #         Xnew, ynew = rotation3D(Xnew, ynew)
        #     elif do[i] == 3:
        #         Xnew, ynew = elastic(Xnew, ynew)
            # if do[i] == 0:
            #     Xnew, ynew = shift3D(Xnew, ynew)
            # elif do[i] == 1:
            #     Xnew, ynew = swirl3D(Xnew, ynew)
        if do[0] == 1:
            Xnew, ynew = flip3D(Xnew, ynew)

        if do[1] == 1:
            Xnew, ynew = brightness(Xnew, ynew)   

        if do[2] == 1:
            Xnew, ynew = rotation3D(Xnew, ynew)

        if do[3] == 1:
            Xnew, ynew = elastic(Xnew, ynew)

        # if do[4] == 1:
        #     Xnew, ynew = shift3D(Xnew, ynew)

        # if do[5] == 1:
        #     Xnew, ynew = swirl3D(Xnew, ynew)
        
        return Xnew, ynew

def aug_batch(Xb, Yb):
    """
    Generate a augmented image batch 
    """
    batch_size = len(Xb)
    newXb, newYb = np.empty_like(Xb), np.empty_like(Yb)
    
    decisions = random_decisions(batch_size)            
    inputs = [(X, y, do) for X, y, do in zip(Xb, Yb, decisions)]
    pool = mp.Pool(processes=8)
    multi_result = pool.starmap(combine_aug, inputs)
    pool.close()
    
    for i in range(len(Xb)):
        newXb[i], newYb[i] = multi_result[i][0], multi_result[i][1]
        
    return newXb, newYb 
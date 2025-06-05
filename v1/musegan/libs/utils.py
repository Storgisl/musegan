"""
Some codes from https://github.com/Newmu/model_code
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio.v2 as imageio
import math
import json
import random
import scipy.misc
import numpy as np
from time import gmtime, strftime
from musegan.libs import write_midi
import sys
import os
import imageio
import glob


get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path, name='result', boarder=True, type_=0):
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = os.path.join(path, name if name.endswith('.png') else name + '.png')

    print(f"[imsave] Input shape: {images.shape}")  # Debug

    # Flatten grid if it's a grid of grids (e.g., (4, 4, 96, 84, 3))
    if images.ndim == 5 and images.shape[:2] == tuple(size):
        images = images.reshape(-1, *images.shape[2:])
    if len(images) > size[0] * size[1]:
        print(f"[imsave] Warning: {len(images)} images won't fit in {size[0]}x{size[1]} grid. Only first {size[0]*size[1]} will be saved.")
    max_images = size[0] * size[1]
    if len(images) > max_images:
        print(f"[imsave] Warning: {len(images)} images won't fit in {size[0]}x{size[1]} grid. Only first {max_images} will be saved.")
        images = images[:max_images]
    merged_image = merge(images, size, boarder=boarder)

    print(f"[imsave] Merged image shape: {merged_image.shape}")  # Debug

    # Ensure image has valid dimensions
    if merged_image.ndim == 4:
        raise ValueError("Merged image is 4D — you probably didn't reshape or merge correctly.")
    elif merged_image.ndim == 2:
        pass  # grayscale
    elif merged_image.ndim == 3 and merged_image.shape[2] not in [1, 3, 4]:
        raise ValueError("Merged image 3rd dimension must be 1 (grayscale), 3 (RGB), or 4 (RGBA).")
    elif merged_image.ndim > 3:
        raise ValueError("Merged image must be 2D or 3D (RGB), but got shape {}".format(merged_image.shape))

    imageio.imwrite(output_path, merged_image)


def merge(images, size, boarder=True):
    """Merge (N, H, W, C) images into one image grid"""
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    boarder = 1 if boarder else 0

    # Create a blank canvas for merged image
    img = np.zeros(((h + boarder) * size[0] - boarder,
                    (w + boarder) * size[1] - boarder, c), dtype=np.uint8)

    for idx, image in enumerate(images):
        i = idx % size[1]  # column
        j = idx // size[1]  # row
        img[j * (h + boarder):j * (h + boarder) + h,
            i * (w + boarder):i * (w + boarder) + w, :] = image
    return img


def to_image_np(bars):
    print(f"[to_image_np] Raw input shape: {bars.shape}")  # (64, 4, 4, 96, 84, 5)
    
    bars = np.clip(bars, 0, 1)
    bars = (bars * 255).astype(np.uint8)

    # Merge tracks horizontally (H, W, T) → (H, W*T)
    bars = bars.reshape(-1, *bars.shape[-3:])  # (N, 96, 84, 5)
    bars = bars.transpose(0, 1, 2, 3)  # (N, H, W, T)
    bars = bars.reshape(bars.shape[0], bars.shape[1], bars.shape[2] * bars.shape[3])  # (N, H, W*T)

    # Add RGB channels
    bars = np.stack([bars] * 3, axis=-1)  # (N, H, W*T, 3)

    print(f"[to_image_np] After reshape: {bars.shape}")
    return bars



def save_bars(bars, size=None, file_path='.', name='sample', type_=0):
    images = to_image_np(bars)
    n_images = images.shape[0]

    if size is None:
        grid_w = int(np.ceil(np.sqrt(n_images)))
        grid_h = int(np.ceil(n_images / grid_w))
        size = [grid_h, grid_w]

    print(f"[save_bars] Saving {n_images} images with grid size {size}")
    return imsave(images, size, file_path, name=name, type_=type_)


def save_midis(bars, file_path):
    # Padding 24 left, 20 right to pitch dimension (axis=4)
    zero_left = np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], bars.shape[3], 24, bars.shape[5]))
    zero_right = np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], bars.shape[3], 20, bars.shape[5]))
    padded_bars = np.concatenate((zero_left, bars, zero_right), axis=4)  # shape: (B, P, B, S, 128, T)

    # Reshape to (samples, step, pitch, track)
    images_with_pause = padded_bars.reshape(-1, padded_bars.shape[3], padded_bars.shape[4], padded_bars.shape[5])

    # Split each track into its own piano roll array
    images_with_pause_list = []
    for ch_idx in range(images_with_pause.shape[3]):  # Iterate over track/channel dimension
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx])

    # Write to MIDI
    write_midi.write_piano_rolls_to_midi(
        images_with_pause_list,
        program_nums=[33, 0, 25, 49, 0],  # GM programs for bass, drums, guitar, piano, strings
        is_drum=[False, True, False, False, False],
        filename=file_path,
        tempo=80.0
    )


def transform(image, npx=64, resize_w=64):
    # npx : # of pixels width/height of image
    return np.array(image)/127.5 - 1.

def make_gif(imgs_filter, gen_dir='./', stop__frame_num=10):
    img_list = glob.glob(imgs_filter)
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    print('%d imgs'% len(img_list))

    stop_frame = np.zeros(images[0].shape)
    images = images + [stop_frame] * stop__frame_num

    imageio.mimsave(os.path.join(gen_dir, 'movie.gif'), images, duration=0.3)


def to_binary_np(bar, threshold=0.0):
    bar_binary = (bar > threshold)
    melody_is_max = (bar[..., 0] == bar[..., 0].max(axis=2, keepdims=True))
    bar_binary[..., 0] = np.logical_and(melody_is_max, bar_binary[..., 0])
    return bar_binary

def to_chroma_np(bar, is_normalize=True):
    chroma = bar.reshape(bar.shape[0], bar.shape[1], 12, 7, bar.shape[3]).sum(axis=3)
    if is_normalize:
        chroma_max = chroma.max(axis=(1, 2, 3), keepdims=True)
        chroma_min = chroma.min(axis=(1, 2, 3), keepdims=True)
        return np.true_divide(chroma + chroma_min, chroma_max - chroma_min + 1e-15)
    else:
        return chroma


def bilerp(a0, a1, b0, b1, steps):

    at = 1 / (steps - 1)
    bt = 1 / (steps - 1)

    grid_list = []
    for aidx in range(0, steps):
        for bidx in range(0, steps):
            a = at * aidx
            b = bt * bidx

            ai = (1-a)*a0 + a*a1
            bi = (1-b)*b0 + b*b1

            grid_list.append((ai, bi))
    return grid_list


def lerp(a, b, steps):
    vec = b - a
    step_vec = vec / (steps+1)
    step_list = []
    for idx in range(1, steps+1):
        step_list.append(a + step_vec*idx)
    return step_list

def slerp(a, b, steps):
    aa =  np.squeeze(a/np.linalg.norm(a))
    bb =  np.squeeze(b/np.linalg.norm(b))
    ttt = np.sum(aa*bb)
    omega = np.arccos(ttt)
    so = np.sin(omega)
    step_deg = 1 / (steps+1)
    step_list = []

    for idx in range(1, steps+1):
        t = step_deg*idx
        tmp = np.sin((1.0-t)*omega) / so * a + np.sin(t*omega)/so * b
        step_list.append(tmp)
    return step_list

def get_sample_shape(sample_size):
    if sample_size >= 64 and sample_size % 8 == 0:
        return [8, sample_size // 8]
    elif sample_size >= 48 and sample_size % 6 == 0:
        return [6, sample_size // 6]
    elif sample_size >= 24 and sample_size % 4 == 0:
        return [4, sample_size // 4]
    elif sample_size >= 15 and sample_size % 3 == 0:
        return [3, sample_size // 3]
    elif sample_size >= 8 and sample_size % 2 == 0:
        return [2, sample_size // 2]
    else:
        # fallback for odd or non-standard sample sizes
        grid_w = int(np.ceil(np.sqrt(sample_size)))
        grid_h = int(np.ceil(sample_size / grid_w))
        return [grid_h, grid_w]


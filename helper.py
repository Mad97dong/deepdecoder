import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread

def load_img(path, img_name):
    img_path = os.path.join(path, img_name + ".png")
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img_clean = img / 255.
    return img_clean

def load_mask(path):
    img_path = os.path.join(path, "mask.png")
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img_clean = img / np.amax(img)
    return img_clean

def psnr(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat - x_true))
    psnr_ = 10. * np.log(maxv ** 2 / mse) / np.log(10.)
    return psnr_


def plot_results(out_img, img_clean, img_noisy):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))

    ax1.imshow(img_clean)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2.imshow(img_noisy)
    ax2.set_title("Noisy observation, PSNR: %.2f" % psnr(img_clean, img_noisy))
    ax2.axis('off')

    ax3.imshow(out_img)
    ax3.set_title("Deep-Decoder denoised image, SNR: %.2f" % psnr(img_clean, out_img))
    ax3.axis('off')
    plt.show()

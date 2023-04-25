import time

import cv2
import numpy as np
from Crypto.Cipher import AES
from matplotlib import pyplot as plt
from skimage.util import img_as_float


def img_encrypt(img, key):
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.encrypt(np.ndarray.tobytes(img))


def img_decrypt(enc_img, key):
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.decrypt(enc_img)


def convert_to_vector(img):
    return img.reshape((img.shape[0] * img.shape[1]))


def convert_from_vector(vector, shape0, shape1):
    img = np.frombuffer(vector, dtype=np.uint8)
    return img.reshape((shape0, shape1))


def plot_img_and_hist(image, axes, bins=256):
    # Преобразование изображения в формат с плавающей запятой двойной точности
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_img.imshow(image, cmap=plt.cm.gray)

    # Display histogram
    ax_hist.hist(image.flatten(), bins=bins, histtype='step', color='black')
    ax_hist.set_xlabel('Pixel intensity', fontsize=25)
    ax_hist.tick_params(axis="x", labelsize=20)
    ax_hist.tick_params(axis="y", labelsize=20)

    return ax_img, ax_hist


key = 'HelloWorldRussia'.encode(encoding='UTF-8')

image = cv2.imread('baboon.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
vector = convert_to_vector(gray)
start = time.time()
encrypted = img_encrypt(vector, key)
img_enc = convert_from_vector(encrypted, gray.shape[0], gray.shape[1])
decrypted = img_decrypt(encrypted, key)
end = time.time() - start
print(end)
img_decr = convert_from_vector(decrypted, gray.shape[0], gray.shape[1])
fig = plt.figure(figsize=(15, 15))
axes = np.zeros((2, 1), dtype=object)
axes[0, 0] = fig.add_subplot(211)
axes[1, 0] = fig.add_subplot(212)
#ax_img, ax_hist = plot_img_and_hist(img_enc, axes[:, 0])
ax_img, ax_hist = plot_img_and_hist(img_decr, axes[:, 0])
ax_hist.set_ylabel('Number of pixels', fontsize=25)
# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()

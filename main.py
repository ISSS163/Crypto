# This is a sample Python script.
import aes
import math
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import skvideo

skvideo.setFFmpegPath('C:/Users/DPOPOV/PycharmProjects/Crypto/ffmpeg-master-latest-win64-gpl/bin/')
import skvideo.io
from skimage.util import img_as_float


# Переписать генерацию ключа в массив
# разобраться почему шифрование - говно (скорее всего проблема с ключом) см generate_key и generate_n

def chebushev(xn, l1, n):
    x = np.zeros(n)
    x[0] = xn
    for i in range(n - 1):
        x[i + 1] = math.cos(l1 * (1 / math.cos(x[i])))
    return x[::-1]


def logistic(x0, l2, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = l2 * x[i] * (1 - x[i])
    return x


def cubic(x0, l3, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = l3 * x[i] * (1 - x[i] ** 2)
    return x


def sine(x0, l4, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = l4 * math.sin(math.pi * x[i])
    return x


def tent(x0, mu, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        if (x[i] <= mu):
            x[i + 1] = x[i] / mu
        else:
            x[i + 1] = (1 - x[i]) / (1 - mu)
    return x


def henon(x0, x1, a, l5, n):
    x = np.zeros(n)
    x[0] = x0
    x[1] = x1
    for i in range(3, n):
        x[i] = 1 + l5 * (x[i - 2] - x[i - 3]) - a * (x[i - 2] ** 2)
    return x


def generate_i_c(key):
    n = 0.0
    for i in key:
        n += int(i, 2) / 256

    return n - math.floor(n)


def generate_key():
    key = []
    for i in range(0, 32):
        key.append(str.ljust(bin(random.randint(1, 255)).replace("0b", ''), 8, '0'))
    return key


def generate_secret_key(bk, lk, ck, sk, tk, hk):
    res = np.zeros((len(bk)))
    for i in range(len(bk)):
        res[i] = int(bk[i], 2) ^ int(lk[i], 2) ^ int(ck[i], 2) ^ int(sk[i], 2) ^ int(tk[i], 2) ^ int(hk[i], 2)
    return res


def generate_n(img):
    vector = img.reshape((img.shape[0] * img.shape[1],))

    b_vector = []
    n = 0

    for i in range(np.size(vector)):
        b_vector.append(str.zfill(bin(vector[i]).replace("0b", ''), 8))
        if i % 32 == 0:
            n += 1
    return n, b_vector


def encrypt(img_list, key, orig):
    res = []

    img_list_int = np.zeros(len(img_list))
    for i in range(len(img_list)):
        img_list_int[i] = int(img_list[i], 2)
    for i in range(int(len(img_list) / len(key))):
        img_cut = img_list_int[i * len(key):i * len(key) + len(key)]
        res.append(np.int_(img_cut) ^ np.int_(key))
    nail = img_list_int[int(len(img_list) / len(key)) * len(key): len(img_list)]
    res.append(np.int_(nail) ^ np.int_(key[:len(nail):]))
    return np.array([x for l in res for x in l]).reshape((orig.shape[0], orig.shape[1]))


def decrypt(encrimg, key):
    res = []
    encr_vector = encrimg.reshape((encrimg.shape[0] * encrimg.shape[1],))
    for i in range(int(len(encr_vector) / len(key))):
        img_cut = encr_vector[i * len(key):i * len(key) + len(key)]
        res.append(np.int_(img_cut) ^ np.int_(key))
    nail = encr_vector[int(len(encr_vector) / len(key)) * len(key): len(encr_vector)]
    res.append(np.int_(nail) ^ np.int_(key[:len(nail):]))
    return (np.array([x for l in res for x in l])).reshape((encrimg.shape[0], encrimg.shape[1]))


def float_to_bin_fixed(f):
    if not math.isfinite(f):
        return repr(f)  # inf nan

    sign = '-' * (math.copysign(1.0, f) < 0)
    frac, fint = math.modf(math.fabs(f))  # split on fractional, integer parts
    n, d = frac.as_integer_ratio()  # frac = numerator / denominator
    assert d & (d - 1) == 0  # power of two
    return f'{sign}{math.floor(fint):b}.{n:0{d.bit_length() - 1}b}'


def convert(arr):
    chaos = []
    for i in arr:
        chaos.append(float_to_bin_fixed(i).replace('-', '')[2:10:])
    return chaos


def calculate_math(enctypted):
    # m_x = np.mean(enctypted, axis=1)
    # m_y = np.mean(enctypted, axis=0)
    # print('m_x', m_x)
    # print('m_y', m_y)
    var = 0
    m = np.mean(enctypted)
    print('m', m)
    for i in range(enctypted.shape[0]):
        for j in range(enctypted.shape[1]):
            var += (enctypted[i][j] - m) ** 2
    var = var / (enctypted.shape[0] * enctypted.shape[1])
    print('var', var)
    # var_x = 1 / (enctypted.shape[0] * enctypted.shape[1]) * np.sum(a)
    # var_y = np.var(enctypted, axis=0)
    # print('var_x', var_x)
    # print('var_y', var_y)

    # con_xy = 1 / (enctypted.shape[0] * enctypted.shape[1]) * np.sum()


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # block size = 256
    start = time.time()
    l1 = 4
    l2 = 4
    l3 = 2.59
    l4 = 0.99
    l5 = 0.3
    a = random.uniform(1.07, 1.09)
    mu = 0.4

    vid_capture = cv2.VideoCapture('video.MOV')
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    savez_dict = dict()
    writer = skvideo.io.FFmpegWriter('output_video.avi', outputdict={
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',  # set the constant rate factor to 0, which is lossless
        '-preset': 'veryslow'  # the slower the better compression, in princple, try
        # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    })
    # output = cv2.VideoWriter('output_video.avi',
    #                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, frame_size)

    cnt = 0
    while (vid_capture.isOpened()):
        # Метод vid_capture.read() возвращает кортеж, первым элементом которого является логическое значение,
        # а вторым - кадр
        ret, frame = vid_capture.read()
        if ret:
            img = np.int_(rgb2gray(frame) * 255)

            n, img_b = generate_n(img)
            key = generate_key()
            ic = generate_i_c(key)
            bk = chebushev(ic, l1, n)
            lk = logistic(ic, l2, n)
            ck = cubic(ic, l3, n)
            sk = sine(ic, l4, n)
            tk = tent(ic, mu, n)
            hk = henon(ic, ic, a, l5, n)

            secret_key = generate_secret_key(convert(bk), convert(lk), convert(ck), convert(sk), convert(tk),
                                             convert(hk))
            savez_dict['arr_%d' % cnt] = secret_key
            encrypted = encrypt(img_b, secret_key, img)
            # decrypted = decrypt(encrypted, secret_key)
            writer.writeFrame(encrypted)
            cnt += 1
        else:
            print('Поток отключен')
            break
    vid_capture.release()
    writer.close()
    np.savez_compressed('keys/key.npz', **savez_dict)
    aes.aes_encrypt("Hello".encode(encoding='UTF-8'))

    vid_capture = cv2.VideoCapture('output_video.avi')
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    writer = skvideo.io.FFmpegWriter('output_decrypted_video.avi', outputdict={
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',  # set the constant rate factor to 0, which is lossless
        '-preset': 'veryslow'  # the slower the better compression, in princple, try
        # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    })
    # output = cv2.VideoWriter('output_decrypted_video.avi',
    #                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, frame_size)
    cnt = 0
    while (vid_capture.isOpened()):
        # Метод vid_capture.read() возвращает кортеж, первым элементом которого является логическое значение,
        # а вторым - кадр
        ret, frame = vid_capture.read()
        if ret:

            img = np.int_(rgb2gray(frame) * 255)
            keys = np.load('keys/key.npz', allow_pickle=True)

            n, img_b = generate_n(img)
            decrypted = decrypt(img, keys['arr_%d' % cnt])
            writer.writeFrame(decrypted)
            cnt += 1
        else:
            print('Поток отключен')
            break
    vid_capture.release()
    writer.close()

    # calculate_math(enctypted)
    #
    # end = time.time() - start
    # print(end)
    #
    # fig = plt.figure(figsize=(15, 15))
    # axes = np.zeros((2, 1), dtype=object)
    # axes[0, 0] = fig.add_subplot(211)
    # axes[1, 0] = fig.add_subplot(212)
    #
    # # ax_img, ax_hist = plot_img_and_hist(enctypted, axes[:, 0])
    # ax_img, ax_hist = plot_img_and_hist(decrypted, axes[:, 0])
    # ax_hist.set_ylabel('Number of pixels', fontsize=25)
    # # prevent overlap of y-axis labels
    # fig.tight_layout()
    # plt.show()
# toDo Реализовать шифрование ключа, шифрование методом AES, сделать извлечение изображения из видео, атаку

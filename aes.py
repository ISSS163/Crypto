import Crypto
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def aes_encrypt(key):
    cnt = 0
    cipher = AES.new(key, AES.MODE_CBC)
    file = open("keys/aes_key.txt", "w")
    with np.load('keys/key.npz') as data:
        for key in data:
            ciphertext = cipher.encrypt(np.ndarray.tobytes(data[key]))
            file.write(str(ciphertext))
            cnt += 1


# def convert_byte(data):

def aes_decrypt(key):
    cipher = AES.new(key, AES.MODE_CBC)
    with open('keys/aes_key.txt', "rb") as file:
        data = np.frombuffer(unpad(cipher.decrypt(pad(file.read(), 16)), 16))

    return data


key = "HelloWorldAndRus".encode(encoding='UTF-8')
aes_encrypt(key)
np.savez_compressed('keys/key.npz', **aes_decrypt(key))

import numpy as np
from Crypto.Cipher import AES


def aes_encrypt(key):
    cnt = 0
    cipher = AES.new(key, AES.MODE_CBC)
    file = open("keys/aes_key.txt", "ab")
    with np.load('keys/key.npz') as data:
        for key in data:
            ciphertext = cipher.encrypt(np.ndarray.tobytes(data[key]))
            file.write(ciphertext)
            cnt += 1


# def convert_byte(data):
# 45331200
def aes_decrypt(key, N):
    cipher = AES.new(key, AES.MODE_CBC)
    with open('keys/aes_key.txt', "rb") as file:
        decrypted = cipher.decrypt(file.read())
        data = np.frombuffer(decrypted, dtype=np.float64)
    dict_data = dict()
    cnt = 0
    for arr in np.split(data, N):
        dict_data["arr_%d" % cnt] = arr
        cnt += 1
    return dict_data



import cv2
import face_recognition
import os
import aes_img
# Импортируем все модули

image = cv2.imread('Lenna.png')

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
vector = aes_img.convert_to_vector(img)
vector_encrypted = aes_img.img_encrypt(vector, 'HelloWorldRussia'.encode('UTF-8'))
image_encrypted = aes_img.convert_from_vector(vector_encrypted, img.shape[0], img.shape[1])
# Методом VideoCapture мы получаем видео, тут можно указать путь к mp4 файлу и прочитать его
# Указав 0 мы получаем видео с ВЕБ КАМЕРЫ

image_enc = face_recognition.face_encodings(img)[0]
# Тут методом face_encodings мы получаем КОДИРОВКУ ЛИЦА рами малека.
# Просто у каждого фото с лицом (да и не только) есть КОДИРОВКА.
# Если у нас есть 2 фото с лицами и если их кодировки совпадают, значит на фото один и тот же человвек

recognizer_cc = cv2.CascadeClassifier('faces.xml')
# Про это уже говорил

recognize = recognizer_cc.detectMultiScale(image_encrypted, scaleFactor=2, minNeighbors=3)

if len(recognize) != 0:
        # Если на фото есть лицо, делаем то, что ниже
    print("Лицо нашел")
    unknown_face = face_recognition.face_encodings(img)
        # Получем кодировку неизвестного лица (лица которое на видео)

    compare = face_recognition.compare_faces([unknown_face], image_enc)
        # Сравниваем две кодировки (кодировку рами малека и кодировку неизвестного лица)
        # Первый параметр надо передать как список (за это и обернули в [])
        # Второй это кодировка рами малека (этот аргумент передаем просто)

    if compare == True:
            # Если мы зашли сюда, значит лица одинаковые
        print('Рами приблизился к вашему дому!')
    else:
        print('Все ок.')
import numpy as np
import cv2
import time
from PIL import ImageOps, Image

def search(image):
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = "C:/Users/1324l/Searching/yolov3.cfg"
    weights_path = "C:/Users/1324l/Searching/yolov3.weights"
    labels = open("C:/Users/1324l/Searching/coco.names").read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    start = time.perf_counter()
    for output in layer_outputs:
        # перебираем все обнаруженные объекты
        for detection in output:
            # извлекаем идентификатор класса (метку) и достоверность (как вероятность)
            # обнаружение текущего объекта
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # отбросим слабые прогнозы, убедившись, что у обнаруженных
            # вероятность больше минимальной вероятности
            if confidence > CONFIDENCE:
                # масштабируем координаты ограничивающего прямоугольника относительно
                # размер изображения, учитывая, что YOLO на самом деле
                # возвращает центральные координаты (x, y) ограничивающего
                # поля, за которым следуют ширина и высота полей
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # используем центральные координаты (x, y) для получения вершины и
                # и левый угол ограничительной рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # обновить наш список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # выполнить не максимальное подавление с учетом оценок, определенных ранее
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    font_scale = 1
    thickness = 1
    arr = []
    # убедитесь, что существует хотя бы один обнаруженный объект
    if len(idxs) > 0:
        # перебираем сохраняемые индексы
        for i in idxs.flatten():
            # извлекаем координаты ограничивающего прямоугольника
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            person = Image.fromarray(np.uint8(image)).crop((x, y, x + w, y + h))
            arr.append(person)
            arr.append([x, y, w, h])
            # шифрование
            # ready_img = image.paste(person, (x, y + h))
            # возврат кадра

            # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
            color = [int(c) for c in colors[0]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[0]}: {confidences[i]:.2f}"
            # вычисляем ширину и высоту текста, чтобы рисовать прозрачные поля в качестве фона текста
            (text_width, text_height) = cv2.getTextSize(text,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        fontScale=font_scale,
                                                        thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # добавить непрозрачность (прозрачность поля)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # теперь поместите текст (метка: доверие%)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            time_took = time.perf_counter() - start
            print("Time took:", time_took)
        if arr is not None:
            return arr

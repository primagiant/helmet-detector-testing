import cv2 as cv
import cvzone
import pandas as pd
from ultralytics import YOLO
import math

from tracker import Tracker


def distance_from_point_to_line(x1, y1, x2, y2, point):
    # Tentukan koefisien A, B, dan C untuk persamaan garis Ax + By + C = 0
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1

    # Hitung jarak dari titik (x3, y3) ke garis
    distance = abs(A * point[0] + B * point[1] + C) / math.sqrt(A ** 2 + B ** 2)

    return distance


def count_object_(frame, bbox_idx, list_arr, offset, label):
    for bbox in bbox_idx:
        x1, y1, x2, y2, cls = bbox
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        d = distance_from_point_to_line(0, 640, 640, 160, (cx, cy))
        if d <= offset:
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cvzone.putTextRect(frame, f'{label}', (x1, y1), 1, 1)
            if list_arr.count(cls) == 0:
                list_arr.append(cls)


video = cv.VideoCapture('./video/test.mp4')

bikerider_model = YOLO('models/bikerider.pt')
helmet_model = YOLO('models/helmet.pt')

bikerider_coco_file = open('models/bikerider_coco.txt', 'r')
bikerider_data = bikerider_coco_file.read()
bikerider_class = bikerider_data.split("\n")

helmet_coco_file = open('models/helmet_coco.txt', 'r')
helmet_data = helmet_coco_file.read()
helmet_class = helmet_data.split("\n")

cy1 = 427
cx1 = 0
offset = 6

other_tracker = Tracker()
bikerider_tracker = Tracker()

no_helmet_tracker = Tracker()
helmet_tracker = Tracker()

other = []
bikerider = []

no_helmet = []
helmet = []

count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv.resize(frame, (640, 640))

    bikerider_results = bikerider_model.predict(frame)
    bikerider_boxes = bikerider_results[0].boxes.data
    bikerider_coors = pd.DataFrame(bikerider_boxes).astype("float")

    bikerider_list = []
    other_list = []
    no_helmet_list = []
    helmet_list = []

    # line visualisation
    cv.line(frame, (0, 640), (640, 160), (0, 255, 255))

    for index, row in bikerider_coors.iterrows():
        b_x1 = int(row[0])
        b_y1 = int(row[1])
        b_x2 = int(row[2])
        b_y2 = int(row[3])

        cls = bikerider_class[int(row[5])]

        if 'bikerider' in cls:
            bikerider_list.append([b_x1, b_y1, b_x2, b_y2])
            # cv.rectangle(frame, (b_x1, b_y1), (b_x2, b_y2), (0, 255, 255), 1)
            # cvzone.putTextRect(frame, f"Bikerider", (b_x1, b_y1), 1, 1)

            cropped_frame = frame[b_y1:b_y2, b_x1:b_x2]
            helmet_results = helmet_model.predict(cropped_frame)
            helmet_boxes = helmet_results[0].boxes.data
            helmet_coors = pd.DataFrame(helmet_boxes).astype("float")

            for i, r in helmet_coors.iterrows():
                x1 = int(b_x1 + r[0])
                y1 = int(b_y1 + r[1])
                x2 = int(b_x1 + r[2])
                y2 = int(b_y1 + r[3])

                cls = helmet_class[int(r[5])]

                if 'nohelmet' in cls:
                    no_helmet_list.append([x1, y1, x2, y2])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cvzone.putTextRect(frame, f'No Helmet', (x1, y1), 1, 1)

                elif 'helmet' in cls:
                    helmet_list.append([x1, y1, x2, y2])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cvzone.putTextRect(frame, f'Helmet', (x1, y1), 1, 1)

    bbox_no_helmet_idx = no_helmet_tracker.update(no_helmet_list)
    count_object_(frame, bbox_no_helmet_idx, no_helmet, offset, "no helmet")

    bbox_helmet_idx = helmet_tracker.update(helmet_list)
    count_object_(frame, bbox_helmet_idx, helmet, offset, "helmet")

    countHelmet = (len(helmet))
    cvzone.putTextRect(frame, f'Helmet:{countHelmet}', (50, 30), 1, 1)

    countNoHelmet = (len(no_helmet))
    cvzone.putTextRect(frame, f'No Helmet:{countNoHelmet}', (50, 60), 1, 1)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

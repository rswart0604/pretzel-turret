import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

if __name__ == '__main__':

    n = 0
    i = 0

    face_stuff = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video = cv2.VideoCapture(0)

    while True:
        list_of_objects = range(n)
        ret, frame = video.read()
        gray_photo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_stuff.detectMultiScale(gray_photo)
        print(faces)
        for (x, y, w, h) in faces:
            # global n
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 2)

        cv2.imshow('photo', frame)
        cv2.imshow('gray', gray_photo)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

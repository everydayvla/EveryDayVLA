import cv2
import numpy as np
import urllib.request as urlreq

import networkx as nx
import os


import random

from math import pi


# import math
# print(math.atan2(0.2, 0.1) * 180/math.pi)
# print(0.1*math.tan(63 * math.pi/180))
def test_camera():
    url = "http://10.5.0.143:4747/"
    url_vid = url + "video"
    cap = cv2.VideoCapture(url_vid)

    getBattery = url + '/battery'  # Ask BAT level
    toggleLED = url + '/cam/1/led_toggle'  # Change LED light ON/OFF

    def cmdSender(cmd):
        ret = ''
        try:
            fp = urlreq.urlopen(cmd)
            ret = fp.read().decode("utf8")
            fp.close()
        except Exception as e:
            print(e)
        return ret

    print(cmdSender(getBattery))
    print((url))    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read frame!")
            break
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(frame.shape)
            break

    cap.release()
    cv2.destroyAllWindows()

def check_video():
    dir = "./datasets/random_placements_smooth"
    folder_dir = dir + "/ball_random_smooth_4"
    url = folder_dir + "/traj.mp4"
    angles_filename = folder_dir + "/end_effector_angles.txt"
    angles_file = open(angles_filename, "r")

    cap = cv2.VideoCapture(url)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read frame!")
            break
        cv2.imshow("Camera", frame)
        print(frame_num)
        print(angles_file.readline())
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        frame_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(frame.shape)
            break

if __name__ == "__main__":
    test_camera()

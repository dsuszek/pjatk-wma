import argparse
from typing import Any
import cv2
import numpy as np
import sys

def parse_arguments()-> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, default=None,
                        help='Path to image that will be processed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    grayscale = False
    corners = False
    video = cv2.VideoCapture(0 if args.video_path is None else args.video_path)
    if not video.isOpened():
        print('Unable to open video.')
        sys.exit()

    while True:
        ret, frame = video.read()
        if not ret:
            sys.exit()
        if corners:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.04)
            dst = cv2.dilate(dst,None)
            frame[dst>0.01*dst.max()]=[0,0,255]
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imshow('Video', frame)
        keycode = cv2.waitKey(10)
        if keycode == ord('q'):
            break
        elif keycode == ord('s'):
            cv2.imwrite('capture.jpg', frame)
        elif keycode == ord('b'):
            grayscale = not grayscale
        elif keycode == ord('c'):
            corners = not corners
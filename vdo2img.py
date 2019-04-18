# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 00:11:22 2018

@author: rein9
"""

#util:  /drives/c/Users/rein9/anaconda3/python vdo2img.py --input "Track1\\track1\\track1_videos\\Loc1_1.mp4" --output "data\Track1\Loc1_1\img"

import cv2
import numpy as np
import sys
import argparse
import time
import os

#add argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,	help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
#ap.add_argument("-y", "--yolo", required=True,
#	help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
#	help="minimum probability to filter weak detections")
#ap.add_argument("-t", "--threshold", type=float, default=0.3,
#	help="threshold when applyong non-maxima suppression")
commonpath = os.path.join(os.getcwd(), '..')
args = vars(ap.parse_args())
print(args["input"])

vs = cv2.VideoCapture(os.path.join(commonpath, args["input"]))
writer = None
(W, H) = (None, None)

#Load video, determine frame number
try:
	prop = cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

nFrmCnt = 0
#grab every frame
while True:
    # read the next frame
    (grabbed, frame) = vs.read()
#    if nFrmCnt == 1:
#        cv2.imshow("First_Frame", frame)
    # if not grabbed, the we have reached the end of the stream
    if not grabbed:
        break
    #get the correct frame dimension
    if W is None and H is None:
        (H,W) = frame.shape[:2]
    #check is output exists
    outputPath = os.path.join(commonpath, args["output"])
    try: 
        os.stat(outputPath)
    except:
        os.makedirs(outputPath)
    frameName = os.path.join(outputPath, ("%06d.jpg" % nFrmCnt))
    print(frameName)
    
    cv2.imwrite(frameName, frame)
    nFrmCnt+=1

print("finished\n", nFrmCnt)
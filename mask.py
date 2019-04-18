# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 01:44:18 2018

@author: rein9
"""

# util:  /drives/c/Users/rein9/anaconda3/python mask.py --input "data\Track1\Loc1_1\img"
#         --output "data\Track1\Loc1_1\detimg" --detection "data\Track1\Loc1_1\det.txt"
#         --yolo "MASK_RCNN\data"--confidence 0.5 --threshold 0.1 --minFrm 0 --maxFrm 1799

import cv2
import numpy as np
import sys
import argparse
import time
import os

#add argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,	help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
ap.add_argument("-d", "--detection", required=True, help="path to detection text")#path to detection result test file
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections") 
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")#default originally is 0.3
ap.add_argument("-lf", "--minFrm", type=int, default=0,
    help="starting frame number")
ap.add_argument("-hf", "--maxFrm", type=int, default=1799,
    help="max frame number")

wrkdir = os.getcwd()
commonpath = os.path.join(os.getcwd(), '..') 
args = vars(ap.parse_args())

# load labels 
labelsPath = os.path.sep.join([commonpath, args["yolo"], "aicity.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print("all Labels: ", LABELS)
# initialize the list of colors to represent each label
np.random.seed(2)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3))

#get the weight from pretrained models
weightsPath = os.path.sep.join([commonpath, args["yolo"], "yolo-voc_final.weights"])
configPath = os.path.sep.join([commonpath, args["yolo"], "yolo-voc.cfg"])

#get the detection text file
detpath = os.path.join(commonpath, args["detection"])
detfile = open(detpath, 'w')# Initialize the file
detfile.close()

#get the output image path, if not exist make dir
outPath  = os.path.join(commonpath, args["output"])
try:
    os.stat(outPath)
except:
    os.makedirs(outPath)

#load object detector trained on coco dataset
print("[INFO], loading yolo model from disk...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net = cv2.dnn.readNet(weightsPath, configPath)


#loading image
nFrmCnt = 0
#grab every frame
while True:
    #check is exists
    imgPath = os.path.join(commonpath, args["input"])
    frameName = os.path.join(imgPath, "%06d.jpg" % nFrmCnt)
        
    try: 
        fh = open(frameName, 'r') 
    except FileNotFoundError: 
        break
    ########################for test pupose
    if nFrmCnt >= 2:
        break
    ###########################
    # reading image
    image = cv2.imread(frameName)
    (W,H) = image.shape[:2]
#    image = cv2.resize(image, (416, 416))
    print("Original image size for %d frame", nFrmCnt, W, H)
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from image and run the forward pass of YOLO detector
    # return: bounding box and probabilities
    # Usage:blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop]]]]]) -> retval
#    blob = cv2.dnn.blobFromImage(image, 1/255.0, (1080,1920), swapRB=True,crop=False)
    blob = cv2.dnn.blobFromImage(image, 1.0, (960,960), swapRB=True,crop=False)
#    cv2.imshow("Blob {}".format(nFrmCnt), blob)
#    cv2.waitKey(200)# window destroys after 100ms
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
 
    #showing timing info 
    print("[INFO] YOLO detect cost {:.6f} seconds".format(end-start))
    
    #initialize a list of bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []
    
    #loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each detection:
        for detection in output:
            #extract the class ID and confidence of the current detection
#            print("detection res", detection[:4])
            scores = list(detection[5:])
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            #check if the detected probability is greater then the min threshold
            if confidence > args["confidence"]:
#                print("confidence for %d", nFrmCnt, confidence, "detected points", detection[0:4])
                #scale the bounding box back relative to the size of the image,
                #PAY ATTENTION: YOLO returns the center of the bounding box
                
                box = detection[0:4] * np.array([W, H, W, H])
#                print("box after correction", list(box))
                (centerX, centerY, width, height) = box.astype("int")
                
                #from centerX, centerY to derive the left corner of bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                #update the list of bounding box coordinates, score and class ID
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                #end of for output in layerOutputs
                
    #apply NMS to suppress week, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    detfile = open(detpath, 'a')# open deteciton file to append
    #check after NMS, how many detections left
    if len(idxs) > 0:
        outframe = os.path.join(outPath, ("%06d.jpg" % nFrmCnt))
        # out_image = copy.deepcopy(image)#deepcopy image for output, might be necessary
        for i in idxs.flatten():
            #extract the bbox
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            print("resized:", x,y,w,h)
            #draw the bounding box and label in the original image
            print(classIDs[i])
            print("total colors", len(COLORS))
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detfile.write("%d,-1,%d,%d,%d,%d,%.3f,-1,-1,-1,%s\n" %(nFrmCnt,x,y,w,h,confidences[i]*100,LABELS[classIDs[i]]))
        cv2.imwrite(outframe, image)
        
    detfile.close()
#    cv2.imshow("Frame {}".format(nFrmCnt), image)
#    cv2.waitKey(100)# window destroys after 100ms
#    cv2.destroyAllWindows()
    nFrmCnt+=1

print("finished\n", nFrmCnt)
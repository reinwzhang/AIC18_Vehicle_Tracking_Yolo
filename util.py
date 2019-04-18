#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:42:29 2019

@author: rwzhang
"""

import numpy as np
import csv
from progressbar import ProgressBar
import pandas as pd

def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score', -1,-1,-1,'class']).

    Args:
        detections

    Returns:
        list: list containing the detections for each frame.
    """
    
    data = []
    if type(detections) is str:
        raw = pd.read_table(detections, sep=',', header=None, names = ['frame',
                            'id', 'x', 'y','w', 'h', 'score', 'color_1', 'color_2', 'color_3','class'])
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)
    raw['x_right'] = raw['x'] +raw['w']
    raw['y_bot'] = raw['y'] + raw['h']
    frames = np.split(raw, np.where(np.diff(raw['frame']))[0]+1, axis=0)#same frame as the same object

    end_frame = len(frames)
    ptr = 0
    pbar = ProgressBar(maxval=end_frame)
    pbar.start()
    for i in range(end_frame):
        dets = []
        if ptr < len(frames) and frames[ptr]['frame'].iloc[0] == i:
            select_cols = ['x','y','x_right','y_bot', 'score', 'class']
            for j in range(len(frames[ptr][select_cols])):
                x1,y1,x2,y2,s,cls = frames[ptr][select_cols].iloc[j]
#                print("frame", ptr, "bbox spec", x1,y1,x2,y2,s)
                dets.append({'bbox': (x1,y1,x2,y2), 'score': s, 'class': cls})
            ptr += 1
        data.append(dets)
        pbar.update(i)
    pbar.finish()
    return data


def save_to_csv(out_path, tracks):
    """
    Saves tracks to a CSV file.

    Args:
        out_path (str): path to output csv file.
        tracks (list): list of tracks to store.
    """

    with open(out_path, "w") as ofile:
        field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz', 'class']

        odict = csv.DictWriter(ofile, field_names)
        id_ = 1
        for track in tracks:
            for i, bbox in enumerate(track['bboxes']):
                row = {'id': id_,
                       'frame': track['start_frame'] + i,
                       'x': np.int(bbox[0]),
                       'y': np.int(bbox[1]),
                       'w': np.int(bbox[2] - bbox[0]),
                       'h': np.int(bbox[3] - bbox[1]),
                       'score': np.round(track['max_score'],2),
                       'wx': -1,
                       'wy': -1,
                       'wz': -1,
                       'class': track['class']}

                odict.writerow(row)
            id_ += 1

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

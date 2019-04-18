# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:34:32 2019

@author: rein9
"""
import numpy as np
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Debugger(object):

    def __init__(self):
        self.detections = []
        self.filename = 'detection.json'

    def store_detected_bounding_boxes(self,boxes,confidence,classID,frame):

        detection = dict()
        detection['frame'] = frame
        # detection['image'] = image
        #added on 01/09
#        detection['boxes'] = []
        detection['tuple'] = []
        
        for i in range(len(boxes)):
            # detection['boxes'].append(box.get_array())
#            detection['boxes'].append()
            #added on 01/09
            detection['tuple'].append({'boxes': boxes[i], 'score': confidence[i], 'class': classID[i]})
            
        self.detections.append(detection)

    def write_detection(self):
        with open(self.filename, 'w') as f:
            json.dump(self.detections, f, cls=NumpyEncoder)


        # with open(filename, "a") as file:
        #     for frame in boxes:
        #         for box in frame:
        #             file.write('%d' % box.x_center)
        #             file.write(',')
        #             file.write('%d' % box.y_center)
        #             file.write(',')
        #             file.write('%d' % box.height)
        #             file.write(',')
        #             file.write('%d' % box.width)
        #             file.write(' ')
        #         file.write('\n')

    def read_detected_bounding_boxes(self):
        with open(self.filename, 'r') as g:
            mydic_restored = json.loads(g.read())
        return mydic_restored

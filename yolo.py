# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
#from utils.tracklet import Tracklet,Track,BoundingBox
#from utils import visualizer
#from tracking import Tracking
from debugger import Debugger

class YOLO(object):
    _defaults = {
#        "model_path": 'model_data/yolo.h5',#from yolov3
#        "anchors_path": 'model_data/yolo_anchors.txt',
#        "classes_path": 'model_data/coco_classes.txt',
        "model_path": 'model_data/yolo-voc_final.h5', #from 2018AICity_TeamUW
        "anchors_path": 'model_data/yolo-voc_anchors.txt',
        "classes_path": 'model_data/aicity.names',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
            print('Loading successful!!!')
        except:
            print('Loading tiny body!!!')
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print('image size', image.size[1])
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        tuple_xywh = []
        classIDs = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box #already corrected
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#            tuple_xywh.append([(top+bottom)//2, (left+right)//2, right - left, bottom - top, score, predicted_class])
            tuple_xywh.append([(top+bottom)//2, (left+right)//2, right - left, bottom - top])
            classIDs.append(predicted_class)
            #print out the data
            print("Detection result for one frame is: ", label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        #print("Original image for detection is: ", image)
        end = timer()
        print("Total time cose for yolo detection for each frame is: ", end - start)
#        update 01/04
#        debug = Debugger()
#        debug.store_detected_bounding_boxes(tuple_xywh,1)
        return image, tuple_xywh, out_scores, classIDs

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    debug = Debugger()
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    frm_num = 0
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if frame is None:
            break
        image = Image.fromarray(frame)
        image,boxes,confidence,classID = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
            debug.store_detected_bounding_boxes(boxes,confidence,classID,frm_num)#detections are stored per frame slicing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frm_num += 1
    debug.write_detection()
    print("Finished Detection on Video, total frames detected:", frm_num)
    yolo.close_session()

# =============================================================================
# 
from time import time
from utils.util import load_mot, iou, save_to_csv
from progressbar import ProgressBar
import pandas as pd

def tracking(video_path, det_path, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         video_path: path to original video
         det_path: path to the final res.txt with the correct ID
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """    
    tracks_active = []
    tracks_finished = []

    debug = Debugger()
    detections = debug.read_detected_bounding_boxes()
    try:
        detections
    except:
        print('Open Error! No Json file found!')
    print("Total frames read from detection.json is: ", len(detections))
#    detfile = open(detpath, 'a')# open deteciton file to append
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
        
    pbar = ProgressBar(maxval=len(detections))
    pbar.start()
#    for frame_num, detections_frame in enumerate(detections):
    for frame_num in range(10):#for debugging purpose
        return_value, frame = vid.read()
        if frame is None:
            break
        image = Image.fromarray(frame)
        #len detections should be the same as frame number
        frame = detections[frame_num]['frame']#frame_number
#        print("frame #: ", frame, "current index:", i)
        detections_frame = detections[frame_num]['tuple']
        
        # update the starting point to be 0 not 1
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                print("current_frame ", frame_num, track['bboxes'])
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['boxes']))
                if iou(track['bboxes'][-1], best_match['boxes']) >= sigma_iou:
                    track['bboxes'].append(best_match['boxes'])
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    #added class info:
                    track['class'] = track['class']
                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['boxes']], 'max_score': det['score'], 'start_frame': frame_num, 'class': det['class']} for det in dets]
        tracks_active = updated_tracks + new_tracks
        pbar.update(frame_num)
    pbar.finish()

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]
    print("Finished Tracking", tracks_finished)
    print('res to be saved to:', det_path)
    save_to_csv(det_path, tracks_finished)
    #reading in all the data again for re-formatting
    temp_df = pd.read_table(det_path, sep = ',', header=None, names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz', 'class'])
    temp_df.sort_values(by=['frame', 'id'], inplace=True)
    open(det_path, 'w').close()
    temp_df.to_csv(det_path, sep = ',', header=None, index = False)

    return tracks_finished
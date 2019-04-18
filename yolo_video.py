# -*- coding: utf-8 -*-
"""
@author: rwzhang
Run YOLO detection on image, detection/tracking on videos.
#util:  
#
#python yolo_video.py --video "2018AICity_TeamUW/Track1/track1/track1_videos/Loc1_1.mp4" --output "2018AICity_TeamUW/data/Track1/Loc1_1" --tracking 1
#python yolo_video.py --image "2018AICity_TeamUW/data/Track1/Loc1_1/img/000000.jpg" --output "2018AICity_TeamUW/data/Track1/Loc1_1" --tracking 0

"""

# =============================================================================
# Package import
import sys,os
import argparse
from yolo import YOLO, detect_video, tracking
from PIL import Image
from tkinter import *
try:
    from tkinter import messagebox
except:
    # Python 2
    import tkMessageBox as messagebox
import time

# =============================================================================
def detect_img(yolo, img):
    while True:
#        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,_,_,_ = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def answer():
    messagebox.showerror("Error", "Sorry, not valid input")

def callback():
    if messagebox.askyesno('Verify', 'Previous Detection Result found, Re-Run detection?'):
        messagebox.showwarning('Yes', 'Please remove the previous file and Re-Run deteciton')
    else:
        messagebox.showinfo('No', 'Quit Re-Running detection')

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('--model', type=str,
                        help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    parser.add_argument('--anchors', type=str,
                        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
    parser.add_argument('--classes', type=str,
                        help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
    parser.add_argument('--gpu_num', type=int,
                        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
    parser.add_argument('--image', nargs='?', type=str, required=False,
                        help='Image input path for image detection mode')
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument("--video", nargs='?', type=str, required=False,
                        default='./path2your_video',help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="", required=True,
                        help = "Output path for detection result")
    parser.add_argument("--tracking", nargs='?', type=int, default=1,
                         help = "Read in video detection output for tracking")
    parser.add_argument('-sl', '--sigma_l', type=float, default=0,
                        help="low detection threshold")
    parser.add_argument('-sh', '--sigma_h', type=float, default=0.5,
                        help="high detection threshold")
    parser.add_argument('-si', '--sigma_iou', type=float, default=0.5,
                        help="intersection-over-union threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=2,
                        help="minimum track length")
    FLAGS = parser.parse_args()
    
    output_path = os.path.expanduser('~/') + FLAGS.output

    if "image" in FLAGS:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "video" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.video + "," + FLAGS.output)
        img_path = os.path.expanduser('~/') + FLAGS.image
        detect_img(YOLO(**vars(FLAGS)), img_path)
    elif "video" in FLAGS :
        video_path = os.path.expanduser('~/') + FLAGS.video
        if not FLAGS.tracking:
            print("Not tracking", video_path)
            if os.path.isfile("detection.json"):
                if messagebox.askokcancel('Attention', 'Please remove the previous file and Re-Run deteciton'):
                    os.rename("detection.json", time.strftime("%H_%M_%S")+"detection.json")
                    print("performing yolo detection on video")
                    detect_video(YOLO(**vars(FLAGS)), video_path, output_path)
            else:
                print("performing yolo detection on video")
                detect_video(YOLO(**vars(FLAGS)), video_path, output_path)
        else: # tracking
            try:
                os.stat("detection.json")
            except:
                "Please perform video detection first"
            det_path = os.path.join(output_path, 'res.txt')
            print("Detection result will be written to: ", det_path)
            detfile = open(det_path, 'w')# Initialize the file
            detfile.close()
            tracking(video_path, det_path, FLAGS.sigma_l, FLAGS.sigma_h, FLAGS.sigma_iou, FLAGS.t_min)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

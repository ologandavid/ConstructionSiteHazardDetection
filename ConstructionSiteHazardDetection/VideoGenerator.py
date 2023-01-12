import cv2
import numpy as np
import argparse
import pathlib
import os
import open3d as o3d
import matplotlib.pyplot as plt
import glob
import ffmpeg

frame_path = 'C:/Users/ologa/Documents/CMU/Semester1/24.678-ComputerVision/FinProj/ConstructionSiteHazardDetection/project-frames/'

def run_frames(video_name):
    """
    Cycles through frames to generate an output video
    Args: 
        video_name (str): Video Name with mp4 extension
    Outputs:
        None
    """
    img_array = []
    for filename in os.scandir(frame_path):
        filename = filename.name
        img = cv2.imread(frame_path + filename)
        height,width,layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'),25.0,size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()    

def video_slice():
    """
    Divides Video into Individual Frames
    Args: 
        None
    Outputs:
        None
    """
    capture = cv2.VideoCapture(path + "trial.mp4")
    frameNr = 0
    while (True):
        success, frame = capture.read()
        if success:
            cv2.imwrite(f"frame_{frameNr}", frame)
        else:
            break
        frameNr = frameNr+1
    capture.release()

if __name__=="__main__":
    #ffmpeg.input(path+'/*.jpg', pattern_type='glob', framerate=25).output('movie.mp4').run()
    run_frames('MAPS.mp4')


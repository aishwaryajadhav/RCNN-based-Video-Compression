import os
import random
import threading
from moviepy.editor import *

output_dir = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\pytorch-image-comp-rnn-master\\test_frames'
parent_dir=""

def generateFramesFromVideos(file):
    vidcap = cv2.VideoCapture(os.path.join(parent_dir, file))
    success, frame = vidcap.read()
    c = 0
    while success:
        cv2.imwrite(os.path.join(output_dir, "frame_"+str(c)+".png") , frame)     # save frame as .png file
        c = c + 1
        success, frame = vidcap.read()

def generateCategorizedDifferenceImages(file):
    vidcap = cv2.VideoCapture(os.path.join(parent_dir, file))

    c = 0
    w=0
    x=0
    y=0
    z=0
    
    #skip first 10 frames
    while c < 10:
        c = c+1
        success, frame1 = vidcap.read()
        
    while success:
        #pick only the next 5 frames per video
        if c > 15:
            break
            
        c = c + 1
        success, frame2 = vidcap.read()
        
        if success:
            diff = frame2 - frame1
            level = diff.mean()
            
            if level <= 15:
                output_file = 'less_fifteen'
            elif level <= 30:
                output_file = 'less_thirty'
            elif level <= 50:
                output_file = 'less_fifty'
            elif level > 50 and level < 200:
                output_file = 'more_fifty'
            else:
                output_file = 'more_two_hundred'
                
            o_dir = os.path.join(output_dir, output_file)
            file_name = os.path.splitext(file)[0]+'_'+str(c)+'.png'
            
            cv2.imwrite(os.path.join(o_dir, file_name) , diff)     
            frame1 = frame2
    
    c = c+1        

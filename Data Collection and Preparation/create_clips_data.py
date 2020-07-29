from moviepy.editor import *
import random
import os
import shutil
import numpy as np

output_dir = 'C:\\Users\\meais\\Data' 
parent_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\raw-videos\\'


def cut_random_video_clips(file, desired_sz = (320, 512)):

    unreadable_videos=[]
    orig_video_name = os.path.join(parent_dir, file)
    try:
        video = VideoFileClip(orig_video_name)
        duration = int(video.duration)
    except:
        unreadable_videos.append(file)
    else:

    #random number of subclips to be sample from any given video clip        
    no = 0
    if(duration < 5):
        no = 1
    elif(duration < 15):
        no = 3
    elif(duration < 40):
        no = 4
    else:
        no = 5
    
    z = random.sample(range(3, duration-1), no)
        
    for q in z:
        clip = video.subclip(q, q+1)
        clip = clip.resize(height = desired_sz[0])
        
        return clip
        try:
            clip.write_videofile("C:\\Users\\meais\\JNotebooks\\Video_Compression\\test\\test_23_"+file+".mp4")
        except:
            unreadable_videos.append(file)
        else:
            generate32x32sizeclips(clip)
            del clip

    del video.reader
    del video


def generate32x32sizeclips(clip, desired_sz, number_clips=20):
        x = random.sample(range(0, desired_sz[0]), number_clips)
        y = random.sample(range(0, desired_sz[1]), number_clips)

        for i in range(number_clips):    
            new_clip = clip.crop( x1 = x[i] , y1 = y[i] , width = 32, height = 32)
            try:
                new_clip.write_videofile("C:\\Users\\meais\\JNotebooks\\Video_Compression\\cropped\\cropped_"+file+"_"+str(j)+str(i)+".mp4")
            except:
                os.remove(file)
                print("Skipping file {}".format(file))
            else:
                del new_clip


def get_video_blocks(p):
    for file in p:
        cut_random_video_clips(file)


def employ_multi_clip_generator():
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    c=0
    for file in os.listdir(parent_dir):
        if c == 0:
            p1.append(file)
        elif c == 1:
            p2.append(file)
        elif c == 2 :
            p3.append(file)
        else:
            p4.append(file)
        
        c = c + 1
        c = c % 4


    t1 = threading.Thread(target=get_video_blocks, args=(p1,)) 
    t2 = threading.Thread(target=get_video_blocks, args=(p2,)) 
    t3 = threading.Thread(target=get_video_blocks, args=(p3,)) 
    t4 = threading.Thread(target=get_video_blocks, args=(p4,)) 
    
    t1.start() 
    t2.start() 
    t3.start() 
    t4.start() 

    t1.join() 
    t2.join() 
    t3.join() 
    t4.join() 

    print("Done!")


employ_multi_clip_generator()
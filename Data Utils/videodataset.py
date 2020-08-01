import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
# import logging

# logging.basicConfig(filename="LoadVideoFile.log", 
#                     format='%(asctime)s %(message)s') 
  
# #Creating an object 
# logger=logging.getLogger() 
  
# #Setting the threshold of logger to DEBUG 
# logger.setLevel(logging.DEBUG) 

class VideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, rootDir, channels = 3, timeDepth = 30, transform=None, xSize = 32, ySize = 32):
        """
		Args:
			rootDir (string): Directory with all the videoes.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
		"""

        videos = []
        for filename in os.listdir(rootDir):
            videos.append('{}'.format(filename))

        self.vds = videos
        self.rootDir = rootDir
        self.channels = channels
        self.timeDepth = timeDepth
        self.xSize = xSize
        self.ySize = ySize
        self.transform = transform

    def __len__(self):
        return len(self.vds)


    def readVideo(self, videoFile):
        # Open the video file
        vidcap = cv2.VideoCapture(videoFile)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = torch.FloatTensor(self.timeDepth, self.channels, self.xSize, self.ySize)
        s = True

        for f in range(self.timeDepth):
            success, frame = vidcap.read()
            if success:
                h, w  = frame.shape[0:2]
                # print(frame.shape)
                if h != 32 or w != 32:
                    # logger.error('Taking zero frame. The frame for video {} has width : {} and height : {}'.format(videoFile, w, h))
                    frame = torch.zeros((3, 32, 32))
                else:
                    # logger.debug('The frame for video {} has width : {} and height : {}'.format(videoFile, w, h))
                    frame = self.transform(frame)
                    
                video[f, :, :, :] = frame

            else:
                s = False
                break

        return video, s

    
    def __getitem__(self, idx):

        videoFile = os.path.join(self.rootDir, self.vds[idx])
        clip, success = self.readVideo(videoFile)
        
        # logger.info('**************************************************************************************************')
        if success:
            return clip

        else:
            print("Error loading Video")
            return torch.FloatTensor(self.timeDepth, self.channels, self.xSize, self.ySize)





import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
import os
import cv2 

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_folder, nt, image_shape,
                 batch_size=4, shuffle=False, seed=None,
                 output_mode='error',
                 data_format='channels_first'):
        # self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        # self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.data_folder = data_folder
        self.nt = nt    # number of timesteps used for sequences in training
        self.batch_size = batch_size
        self.data_format = data_format
       
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        self.video_clips = np.array([name for name in os.listdir(self.data_folder)])

        # if self.data_format == 'channels_first':
        #     self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = image_shape
        self.im_shape = (self.im_shape[0], self.im_shape[1], self.im_shape[2])
        # if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
        #     self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        # elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
        #     curr_location = 0
        #     possible_starts = []
        #     while curr_location < self.X.shape[0] - self.nt + 1:
        #         if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
        #             possible_starts.append(curr_location)
        #             curr_location += self.nt
        #         else:
        #             curr_location += 1
        #     self.possible_starts = possible_starts

        if shuffle:
            np.random.shuffle(self.video_clips)
        # if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
        #     self.possible_starts = self.possible_starts[:N_seq]
        
        self.N_sequences = len(self.video_clips)
        super(SequenceGenerator, self).__init__(len(self.video_clips), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            clip_file = self.video_clips[idx]
            vidcap = cv2.VideoCapture(os.path.join(self.data_folder,clip_file))
            
            for f in range(self.nt):
                success, frame = vidcap.read()
                if success:
                    if self.data_format == 'channels_first':
                        frame = np.transpose(frame, (2, 0, 1))
                    
                    if frame.shape == self.im_shape:
                        frame = self.preprocess(frame)    
                        batch_x[i, f, 2, :, :] = frame[0]
                        batch_x[i, f, 1, :, :] = frame[1]
                        batch_x[i, f, 0, :, :] = frame[2]
                    else:
                        print(frame.shape)
                        print('Not Ok')

        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255 

    def create_all(self):
        
        X_all = np.empty((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i in range(self.N_sequences):
            clip_file = self.video_clips[i]
            vidcap = cv2.VideoCapture(os.path.join(self.data_folder,clip_file))
            
            for f in range(self.nt):
                success, frame = vidcap.read()
                if success:
                    if self.data_format == 'channels_first':
                        frame = np.transpose(frame, (2, 0, 1))
                    
                    if frame.shape == self.im_shape:
                        frame = self.preprocess(frame)    
                        X_all[i, f, 2, :, :] = frame[0]
                        X_all[i, f, 1, :, :] = frame[1]
                        X_all[i, f, 0, :, :] = frame[2]
                    else:
                        print(frame.shape)
                        print('Not Ok')

        return X_all, self.video_clips



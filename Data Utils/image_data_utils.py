import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
import os
import cv2 

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_folder, image_shape,
                 batch_size=4, shuffle=False, seed=None,
                 data_format='channels_first', N_seq = None):
        # self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        # self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.data_format = data_format
       
        self.img_files = np.array([name for name in os.listdir(self.data_folder)])
        self.im_shape = image_shape
        height = image_shape[1]
        width = image_shape[2]

        self.encoder_h_1 = np.zeros((self.batch_size, 2, 256, height // 4, width // 4), np.float32)
        self.encoder_h_2 = np.zeros((self.batch_size, 2, 512, height // 8, width // 8), np.float32)
        self.encoder_h_3 = np.zeros((self.batch_size, 2, 512, height // 16, width // 16), np.float32)

        self.decoder_h_1 = np.zeros((self.batch_size, 2, 512, height // 16, width // 16), np.float32)
        self.decoder_h_2 = np.zeros((self.batch_size, 2, 512, height // 8, width // 8), np.float32)
        self.decoder_h_3 = np.zeros((self.batch_size, 2, 256, height // 4, width // 4), np.float32)
        self.decoder_h_4 = np.zeros((self.batch_size, 2, 128, height // 2, width // 2), np.float32)


        if shuffle:
            np.random.shuffle(self.img_files)
        
        if N_seq is not None:
            self.img_files = self.img_files[:N_seq]
            self.N_seq = N_seq
        else:
            self.N_seq = len(self.img_files)

        super(SequenceGenerator, self).__init__(len(self.img_files), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        
        batch_x = np.zeros((current_batch_size,) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            img_file = self.img_files[idx]
            image = self.preprocess(cv2.imread(os.path.join(self.data_folder, img_file)))
            
            image = np.transpose(image, (2, 0, 1))
             
            if image.shape != self.im_shape:
                image = np.zeros(self.im_shape, np.float32)
                print('******************Image size does not match!!!*******************')
            batch_x[i,:,:,:] = image
        
        batch_y = np.zeros((current_batch_size,1), np.float32)
        
        return [batch_x, self.encoder_h_1, self.encoder_h_2, self.encoder_h_3, self.decoder_h_1, self.decoder_h_2, self.decoder_h_3, self.decoder_h_4], batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255 - 0.5


    def create_all(self):
        X_all = np.zeros((self.N_seq,) + self.im_shape, np.float32)
        for i in range(self.N_seq):
            image = self.preprocess(cv2.imread(os.path.join(self.data_folder, self.img_files[i])))
            image = np.transpose(image, (2, 0, 1))
           
            if image.shape != self.im_shape:
                image = np.zeros(self.im_shape, np.float32)
                print('******************Image size does not match!!!*******************')

            
            X_all[i] = image
        return X_all

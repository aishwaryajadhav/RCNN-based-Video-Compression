import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, Add
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from compression_model import Encoder, Binarizer, Decoder
from image_data_utils import SequenceGenerator
# from youtube_settings import *

def difference_images(x):
    return x[0]-x[1]

WEIGHTS_DIR='C:\\Users\\meais\\JNotebooks\\Video_Compression\\prednet\\model_data_keras2\\youtube'
train_folder = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\Frames'
val_folder = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\val_frames'

save_model = True  # if weights will be saved
weights = os.path.join(WEIGHTS_DIR, 'encoder_weights.hdf5') 
weights_file = os.path.join(WEIGHTS_DIR, 'encoder_weights_1.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'encoder_model.json')


nb_epoch = 200
batch_size = 8
samples_per_epoch = 4000
N_seq_val = 96  # number of sequences to use for validation


f = open(json_file, 'r')
json_string = f.read()
f.close()
model = model_from_json(json_string, custom_objects = {'Encoder': Encoder, 'Decoder': Decoder, 'Binarizer':Binarizer})
model.load_weights(weights)

n_channels, height, width = (3, 32, 32)    
input_shape = (n_channels, height, width) #if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)

# encoder = Encoder(weights=model.layers[4].get_weights())
# binarizer = Binarizer(weights=model.layers[5].get_weights())
# decoder = Decoder(weights=model.layers[10].get_weights())

# del(model)

# Model parameters

# iterations = 16

# # encoder = Encoder()
# # binarizer = Binarizer()
# # decoder = Decoder()


# image = Input(shape=input_shape)

# encoder_h_1 = Input(shape = (2, 256, height // 4, width // 4))
# encoder_h_2 = Input(shape = (2, 512, height // 8, width // 8))
# encoder_h_3 = Input(shape = (2, 512, height // 16, width // 16))

# decoder_h_1 = Input(shape = (2, 512, height // 16, width // 16))
# decoder_h_2 = Input(shape = (2, 512, height // 8, width // 8))
# decoder_h_3 = Input(shape = (2, 256, height // 4, width // 4))
# decoder_h_4 = Input(shape = (2, 128, height // 2, width // 2))

# # #remember to -0.5 from image 
# res, e1, e2, e3 = image, encoder_h_1, encoder_h_2, encoder_h_3
# d1, d2, d3, d4 = decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4


# for i in range(iterations):
#     [encoded, e1, e2, e3] = encoder([res, e1, e2, e3])
#     coded = binarizer(encoded)
#     [output, d1, d2, d3, d4] = decoder([coded, d1, d2, d3, d4])
#     res = Lambda(difference_images)([image, output])

#     layer_error = Lambda(lambda x: K.mean(K.abs(K.batch_flatten(x)), axis=-1, keepdims=True))(res)
#     if i ==0:
#         all_error = layer_error 
#     else:
#         all_error = Add()([all_error, layer_error])

# # # print('all_error:{}'.format(all_error))

# # losses = Lambda(lambda x: x/iterations)(all_error)

# model = Model(inputs=[image, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4], outputs=all_error)

model.compile(loss='mean_absolute_error', optimizer='adam')  

model.summary()


# if save_model:
#     json_string = model.to_json()
#     with open(json_file, "w") as f:
#         f.write(json_string)


train_generator = SequenceGenerator(train_folder, input_shape, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_folder, input_shape, batch_size=batch_size)


lr_schedule = lambda epoch: 0.0005 if epoch < 50 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1, save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

# if save_model:
#     json_string = model.to_json()
#     with open(json_file, "w") as f:
#         f.write(json_string)




'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
'''
#python youtube_tp1_fine_tune.py
#model json changes: batch_input_shape and backend from Theano to tensorflow
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed

from keras.models import Model, model_from_json
from keras.layers import Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from new_data_utils import SequenceGenerator
from youtube_settings import *

# extrap_start_time = 10  # starting at this time step, the prediction from the previous time step will be treated as the actual input
orig_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-youtubetp1finetuned_2.hdf5')  # original t+1 weights
orig_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
print('Weights file received')
save_model = True
extrap_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-youtubetp1finetuned_1.hdf5')  # where new weights will be saved
extrap_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model-youtubetp1finetuned.json')

# Data files
# train_folder=''
# val_folder=''

nt = 10

# Training parameters
nb_epoch = 100
batch_size = 1
samples_per_epoch = 500
N_seq_val = 40  # number of sequences to use for validation

# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)
print('Model loaded')
im_height, im_width, n_channels = (288, 256, 3)  

layer_config = orig_model.layers[1].get_config()
# layer_config['output_mode'] = 'error'
# layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']

if data_format == 'channels_first':
    input_shape = (nt, n_channels, im_height, im_width) 
else:
    input_shape = (nt, im_height, im_width, n_channels) 

layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

# input_shape = list(orig_model.layers[0].batch_input_shape[1:])
# input_shape[0] = nt

inputs = Input(input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')  #loss is calculated between the outputs and ys | here the ys are 0 (data generator returns 0 in error mode), so loss = mae(errors returned)


train_generator = SequenceGenerator(train_folder, nt, image_shape = input_shape[1:], batch_size=batch_size, shuffle=True, data_format=data_format)
val_generator = SequenceGenerator(val_folder, nt, image_shape = input_shape[1:], batch_size=batch_size, data_format=data_format)

lr_schedule = lambda epoch: 0.0001 if epoch < 75 else 0.00005    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', verbose=1, save_best_only=True))   #Save the model after every epoch.
print('Fit start')
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)
print('fit done')
if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)

'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
'''
#model json changes: batch_input_shape and backend from Theano to tensorflow
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from prednet_compress import PredNet
from new_data_utils import SequenceGenerator
from youtube_settings import *


nt = 10

# Define loss as MAE(mean abs error) of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

orig_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # original t+1 weights
orig_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# orig_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-youtubetp1finetuned.hdf5')  # original t+1 weights
# orig_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model-youtubetp1finetuned.json')

save_model = True
extrap_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_compress_weights-extrapfinetuned.hdf5')  # where new weights will be saved
extrap_json_file = os.path.join(WEIGHTS_DIR, 'prednet_compress_model-extrapfinetuned.json')


# Training parameters
nb_epoch = 150
batch_size = 1
samples_per_epoch = 500
N_seq_val = 40  # number of sequences to use for validation

# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)

im_height, im_width, n_channels = (288, 256, 3)  

layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'

data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

if data_format == 'channels_first':
    input_shape = (nt, n_channels, im_height, im_width) 
else:
    input_shape = (nt, im_height, im_width, n_channels) 

inputs = Input(input_shape)
predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')
model.summary()

train_generator = SequenceGenerator(train_folder, nt, image_shape = input_shape[1:], batch_size=batch_size, shuffle=True, data_format=data_format, output_mode='prediction')
val_generator = SequenceGenerator(val_folder, nt, image_shape = input_shape[1:], batch_size=batch_size, data_format=data_format, output_mode='prediction')

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', verbose=1, save_best_only=True))
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)

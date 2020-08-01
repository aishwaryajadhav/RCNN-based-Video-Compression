
import imageio
import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from metric_image_video import Metric

from prednet import PredNet
from new_data_utils import SequenceGenerator
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\pytorch-image-comp-rnn-master')

from pred_encoder import EncoderImage

n_plot = 40
batch_size = 1
nt = 10
word = 'jockey'

WEIGHTS_DIR='C:\\Users\\meais\\JNotebooks\\Video_Compression\\prednet\\model_data_keras2\\youtube'

input_shape = (nt, 3, 256, 480)
test_folder = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\show\\t'


weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-youtubetp1finetuned_1.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model-youtubetp1finetuned.json')
# test_file = os.path.join(DATA_DIR, 'X_test.hkl')
# test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

enoder_model = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\pytorch-image-comp-rnn-master\\checkpoint\\base_model\\encoder_epoch_00000004.pth'

RESULTS_SAVE_DIR = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\show\\test'
PRED_RESULTS_SAVE_DIR = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\show\\pred'


encoder = EncoderImage(enoder_model, iterations = 16, cuda = False)

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
# input_shape = list(train_model.layers[0].batch_input_shape[1:])

# input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator =  SequenceGenerator(test_folder, nt, image_shape = input_shape[1:], batch_size=4, data_format=data_format)

X_test_all, files = test_generator.create_all()

for i, f in enumerate(files):
    X_test = np.expand_dims(X_test_all[i], axis = 0)
    X_hat = test_model.predict(X_test, batch_size)

    # K.clear_session()
    # del test_model
    # del train_model
    
    X_pred = np.transpose(np.squeeze(X_hat.clip(0, 1) * 255.0).astype(np.uint8) , (0, 2, 3 , 1))
    imageio.mimwrite(os.path.join(PRED_RESULTS_SAVE_DIR,f+'.avi'), X_pred , fps = 30)

    x_shape = X_hat.shape
    res = X_test - X_hat
    res = res.reshape((x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4]))


    for i in range(nt):
        res[i] = np.squeeze(encoder.encode(np.expand_dims(res[i], axis= 0), (64, 64), iterations=16))


    X_hat = X_hat + res.reshape(x_shape)

    # *********You always store in system in the channels last format
    if data_format == 'channels_first':
        # X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    # X_test = (X_test.clip(0, 1) * 255.0).astype(np.uint8)
    X_hat = (X_hat.clip(0, 1) * 255.0).astype(np.uint8)

    # imageio.mimwrite(os.path.join(RESULTS_SAVE_DIR,'X_test_'+word+'.avi'), X_test[0] , fps = 30)
    imageio.mimwrite(os.path.join(RESULTS_SAVE_DIR, f+'.avi'), X_hat[0] , fps = 30)






##########################################################################################
# x_shape = X_hat.shape
# res = X_test - X_hat
# res = res.reshape((x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4]))


# for i in range(nt):
#     res[i] = np.squeeze(encoder.encode(np.expand_dims(res[i], axis= 0), (64,64)))


# X_hat = X_hat + res.reshape(x_shape)

#*********You always store in system in the channels last format
# if data_format == 'channels_first':
#     X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
#     X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# X_test = (X_test.clip(0, 1) * 255.0).astype(np.uint8)
# X_hat = (X_hat.clip(0, 1) * 255.0).astype(np.uint8)

# imageio.mimwrite(os.path.join(RESULTS_SAVE_DIR,'X_test_'+word+'.avi'), X_test[0] , fps = 30)
# imageio.mimwrite(os.path.join(RESULTS_SAVE_DIR,'X_hat_'+word+'.avi'), X_hat[0] , fps = 30)


##########################################################################################



# res = encoder.encode(res,(32,32))
# res[:10] = encoder.encode(res[:10],(32,32))
# res[10:] = encoder.encode(res[10:],(32,32))
# res = encoder.encode(res,(512,960))



# if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
# f = open(os.path.join(RESULTS_SAVE_DIR , 'metric_scores_hd.txt'), 'a')

# (mssim, psnr, mse) = Metric.calc_metric_arrays(X_test[0], X_hat[0], nt)
# f.write("\nBeauty Orig and PredEncoder recon (iter = 16) MSSIM: %f\n" % mssim)
# f.write("Beauty Orig and PredEncoder recon (iter = 16) PSNR: %f\n" % psnr)
# f.write("Beauty Orig and PredEncoder recon (iter = 16) MSE: %f\n" % mse)

# f.close()














# mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
# mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
# mse_pred = np.mean( (X_test[0, 1:] - X_pred[:, 1:])**2 )  # look at all timesteps except the first


# here we are taking MSEs between the prev frame and current frame as well --> a concept on which most of my previous diff models were based -- can you use these differences in the MSEs to show why encoding the first diff (predicted - actual) is better than predicting the second diff (prev - actual/current), thus discussing failed models --> lesser diff/MSE, lesser entropy/variations to encode --> less no. of bits required to encode the differences.

# if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
# f = open(os.path.join(RESULTS_SAVE_DIR , 'prediction_encoder_scores_hd.txt'), 'w')

# f.write("Model MSE between actual and reconstructed: %f\n" % mse_model)
# # f.write("MSE between actual and predicted: %f\n" % mse_pred)
# f.write("Previous Frame MSE: %f" % mse_prev)
# f.close()

#plotting the actual and predicted images in the video
# Plot some predictions

# aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
# plt.figure(figsize = (nt, 2*aspect_ratio))
# gs = gridspec.GridSpec(2, nt)
# gs.update(wspace=0., hspace=0.)
# plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots_t/')
# if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
# plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
# for i in plot_idx:
#     for t in range(nt):
#         plt.subplot(gs[t])
#         plt.imshow(X_test[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Actual', fontsize=10)

#         plt.subplot(gs[t + nt])
#         plt.imshow(X_hat[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Predicted', fontsize=10)

#     plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#     plt.clf()


import os
import numpy as np
from six.moves import cPickle

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input
from compression_model import Encoder, Binarizer, Decoder


# model = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\prednet\\model_data_keras2\\youtube\\encoder_model.json'
# weights = 'C:\\Users\\meais\\JNotebooks\\Video_Compression\\prednet\\model_data_keras2\\youtube\\encoder_weights.hdf5'
# iterations = 16



class Compressor:
    def __init__(self, model=None, weights=None, input_shape=(3,32,32), iterations=16):
        self.iterations = iterations
        self.input_shape = input_shape
        f = open(model, 'r')
        json_string = f.read()
        f.close()
        train_model = model_from_json(json_string, custom_objects = {'Encoder': Encoder, 'Decoder': Decoder, 'Binarizer':Binarizer})
        train_model.load_weights(weights)

        layer_config = train_model.layers[4].get_config()
        layer_config['trainable'] = False
        encoder = Encoder(weights=train_model.layers[4].get_weights())

        layer_config = train_model.layers[5].get_config()
        layer_config['trainable'] = False
        binarizer = Binarizer(weights=train_model.layers[5].get_weights())

        layer_config = train_model.layers[10].get_config()
        layer_config['trainable'] = False
        decoder = Decoder(weights=train_model.layers[10].get_weights())

        del(train_model)
        height = input_shape[-2]
        width = input_shape[-1]
        
        image = Input(shape=input_shape)
        
        encoder_h_1 = Input(shape = (2, 256, height // 4, width // 4))
        encoder_h_2 = Input(shape = (2, 512, height // 8, width // 8))
        encoder_h_3 = Input(shape = (2, 512, height // 16, width // 16))

        decoder_h_1 = Input(shape = (2, 512, height // 16, width // 16))
        decoder_h_2 = Input(shape = (2, 512, height // 8, width // 8))
        decoder_h_3 = Input(shape = (2, 256, height // 4, width // 4))
        decoder_h_4 = Input(shape = (2, 128, height // 2, width // 2))

        #***********************remember to -0.5 from image***************************************************
 
        # res, e1, e2, e3 = image, encoder_h_1, encoder_h_2, encoder_h_3
        # d1, d2, d3, d4 = decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4

        # for i in range(iterations):
            #idea for variable rate /iterations: keep this in the model and delete the loop. Put this loop around model.predict()
        
        [encoded, e1, e2, e3] = encoder([image, encoder_h_1, encoder_h_2, encoder_h_3])
        coded = binarizer(encoded)
        [output, d1, d2, d3, d4] = decoder([coded, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4])
        # res = Lambda(difference_images)([image, output])

        self.model = Model(inputs=[image, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4], outputs=[output, e1, e2, e3, d1, d2, d3, d4])


    def encode(self, image, batch_size = 1):
        image = image - 0.5
        height = self.input_shape[1]
        width = self.input_shape[2]
        res = image
        
        encoder_h_1 = np.zeros((batch_size, 2, 256, height // 4, width // 4), np.float32)
        encoder_h_2 = np.zeros((batch_size, 2, 512, height // 8, width // 8), np.float32)
        encoder_h_3 = np.zeros((batch_size, 2, 512, height // 16, width // 16), np.float32)

        decoder_h_1 = np.zeros((batch_size, 2, 512, height // 16, width // 16), np.float32)
        decoder_h_2 = np.zeros((batch_size, 2, 512, height // 8, width // 8), np.float32)
        decoder_h_3 = np.zeros((batch_size, 2, 256, height // 4, width // 4), np.float32)
        decoder_h_4 = np.zeros((batch_size, 2, 128, height // 2, width // 2), np.float32)

        output_images = np.zeros((batch_size,self.iterations) + self.input_shape, np.float32)

        

        for i in range(self.iterations):
            [output, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4] = self.model.predict([res, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4], batch_size)
            
            res = image - output
            
            output_images[:,i,:,:,:]=  output + 0.5
            # losses.append(np.mean(res**2))

        return output_images






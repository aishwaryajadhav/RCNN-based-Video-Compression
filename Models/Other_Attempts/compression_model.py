import numpy as np

from keras import backend as K
from keras import activations
from keras.engine import InputSpec
from keras.layers import Layer, Lambda
from keras.layers.convolutional import Conv2D
from keras_conv_lstm_layers import ConvLSTM2DCell
#when hidden_filter_size=3 --> padding = same
#when hidden_filter_size=1 --> padding = valid


class Encoder(Layer):

    def __init__(self, data_format='channels_first', **kwargs):
        self.data_format = data_format
        super(Encoder, self).__init__(**kwargs)
        # self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape): #input_shape = [shape, (shape, shape), (shape, shape), (shape, shape)]
        # self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1]), InputSpec(shape=input_shape[2]), InputSpec(shape=input_shape[3])]   #input_shape
        # print('In compression_model build: input_shape: {}'.format(input_shape))
        self.conv =Conv2D(64, 3, strides=2, padding='same', data_format=self.data_format, use_bias=False)

        self.rnn1 = ConvLSTM2DCell(256, 3, 1, strides = 2, padding = 'same', hidden_padding ='valid', use_bias=False, data_format=self.data_format)

        self.rnn2 = ConvLSTM2DCell(512, 3, 1, strides = 2, padding = 'same', hidden_padding = 'valid', use_bias=False, data_format=self.data_format)

        self.rnn3 = ConvLSTM2DCell(512, 3, 1, strides = 2, padding = 'same', hidden_padding = 'valid', use_bias=False, data_format=self.data_format)

        self.trainable_weights = []
        
        i_shape = input_shape[0]
        self.height = i_shape[-2]
        self.width = i_shape[-1]
        

        with K.name_scope('conv'):
            self.conv.build(i_shape)
        self.trainable_weights += self.conv.trainable_weights
         
        i_s = (i_shape[0], 64, self.height // 2, self.width // 2)
        with K.name_scope('rnn1'):
            self.rnn1.build([i_s, input_shape[1]])
        self.trainable_weights += self.rnn1.trainable_weights 
        
        i_s = (i_shape[0], 256, self.height // 4, self.width // 4)
        with K.name_scope('rnn2'):
            self.rnn2.build([i_s, input_shape[2]])
        self.trainable_weights += self.rnn2.trainable_weights 
    
        i_s = (i_shape[0], 512, self.height // 8, self.width // 8)
        with K.name_scope('rnn3'):
            self.rnn3.build([i_s, input_shape[3]])
        self.trainable_weights += self.rnn3.trainable_weights


    def call(self, inputs):
        # print('In call of compression_model:: input: {}, hidden1:{}, hidden2:{}, hidden3:{}'.format(inputs[0], inputs[1],inputs[2], inputs[3]))

        x = self.conv(inputs[0])
        
        hidden1 = self.rnn1([x, inputs[1]])
        x = hidden1[:, 0,:,:,:]
        # print('In network encoder:: x: {}, hidden1:{}'.format(x, hidden1))

        hidden2 = self.rnn2([x, inputs[2]])
        x = hidden2[:, 0,:,:,:]
        # print('In network encoder:: x: {}, hidden2:{}'.format(x, hidden2))

        hidden3 = self.rnn3([x, inputs[3]])
        x = hidden3[:, 0,:,:,:]
        # print('In network encoder:: x: {}, hidden3:{}'.format(x, hidden3))

        return [x, hidden1, hidden2, hidden3]



    def compute_output_shape(self, input_shape):
        i_s = (input_shape[0][0], 512, self.height // 16, self.width // 16)
        return [i_s, input_shape[1], input_shape[2], input_shape[3]]



    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Binarizer(Layer):

    def __init__(self, data_format='channels_first', **kwargs):
        self.data_format = data_format
        super(Binarizer, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        print('In binarizer: input_shape:{}'.format(input_shape))
        self.input_spec = [InputSpec(shape=input_shape)]

        self.conv =Conv2D(32, 1, data_format=self.data_format, use_bias=False, activation='tanh')

        self.trainable_weights = []

        with K.name_scope('bconv'):
            self.conv.build(input_shape)
        self.trainable_weights += self.conv.trainable_weights

    def call(self, inputs):
        return K.sign(self.conv(inputs))
        

    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[0], 32, input_shape[2], input_shape[3])
        return out_shape


    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Binarizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def depth_to_space_keras(x):
        import tensorflow as tf
        return tf.depth_to_space(x, block_size=2, data_format='NCHW')


class Decoder(Layer):

    def __init__(self, data_format='channels_first', **kwargs):
        self.data_format = data_format
        super(Decoder, self).__init__(**kwargs)
        # self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4)]

    
    
    def build(self, input_shape):
        # print('In decoder compression model build: input_shape: {}'.format(input_shape))
        
        self.conv1 =Conv2D(512, 1, strides=1, padding='valid', data_format=self.data_format, use_bias=False)

        self.rnn1 = ConvLSTM2DCell(512, 3, 1, strides = 1, padding = 'same', hidden_padding ='valid', use_bias=False, data_format=self.data_format)

        self.rnn2 = ConvLSTM2DCell(512, 3, 1, strides = 1, padding = 'same', hidden_padding = 'valid', use_bias=False, data_format=self.data_format)

        self.rnn3 = ConvLSTM2DCell(256, 3, 3, strides = 1, padding = 'same', hidden_padding = 'same', use_bias=False, data_format=self.data_format)

        self.rnn4 = ConvLSTM2DCell(128, 3, 3, strides = 1, padding = 'same', hidden_padding = 'same', use_bias=False, data_format=self.data_format)

        self.conv2 =Conv2D(3, 1, strides=1, padding='valid', data_format=self.data_format, use_bias=False)

        self.trainable_weights = []
        
        i_shape = input_shape[0]
        self.height = i_shape[2]
        self.width = i_shape[3]
        

        with K.name_scope('dconv1'):
            self.conv1.build(i_shape)
        self.trainable_weights += self.conv1.trainable_weights
        

        i_s = (i_shape[0], 512, self.height, self.width)
        with K.name_scope('drnn1'):
            self.rnn1.build([i_s, input_shape[1]])
        self.trainable_weights += self.rnn1.trainable_weights 
        
        i_s = (i_shape[0], 128, self.height*2, self.width*2)
        with K.name_scope('drnn2'):
            self.rnn2.build([i_s, input_shape[2]])
        self.trainable_weights += self.rnn2.trainable_weights 
        

        i_s = (i_shape[0], 128, self.height*4, self.width*4)
        with K.name_scope('drnn3'):
            self.rnn3.build([i_s, input_shape[3]])
        self.trainable_weights += self.rnn3.trainable_weights 
        

        i_s = (i_shape[0], 64, self.height*8, self.width*8)
        with K.name_scope('drnn4'):
            self.rnn4.build([i_s, input_shape[4]])
        self.trainable_weights += self.rnn4.trainable_weights 
        
        i_s = (i_shape[0], 32, self.height*16, self.width*16)
        with K.name_scope('dconv2'):
            self.conv2.build(i_s)
        self.trainable_weights += self.conv2.trainable_weights
        



    def call(self, inputs):
        # print('In call of compression_model decoder:: input: {}, hidden1:{}, hidden2:{}, hidden3:{}, hidden4:{}'.format(inputs[0], inputs[1],inputs[2], inputs[3], inputs[4]))

        x = self.conv1(inputs[0])

        hidden1 = self.rnn1([x, inputs[1]])
        x = hidden1[:, 0,:,:]
        x = Lambda(depth_to_space_keras, name='d_2_s0')(x)
        # print('In decoder network:: x: {}, hidden1:{}'.format(x, hidden1))

        hidden2 = self.rnn2([x, inputs[2]])
        x = hidden2[:, 0,:,:]
        x = Lambda(depth_to_space_keras, name='d_2_s1')(x)
        # print('In decoder network:: x: {}, hidden2:{}'.format(x, hidden2))

        hidden3 = self.rnn3([x, inputs[3]])
        x = hidden3[:, 0,:,:]
        x = Lambda(depth_to_space_keras, name='d_2_s2')(x)
        # print('In decoder network:: x: {}, hidden3:{}'.format(x, hidden3))

        hidden4 = self.rnn4([x, inputs[4]])
        x = hidden4[:, 0,:,:]
        x = Lambda(depth_to_space_keras, name='d_2_s3')(x)
        # print('In decoder network:: x: {}, hidden4:{}'.format(x, hidden4))

        x = K.tanh(self.conv2(x)) / 2
        # print('In decoder network:: x: {}'.format(x))

        return [x, hidden1, hidden2, hidden3, hidden4]


    def compute_output_shape(self, input_shape):
        i_s = (input_shape[0][0], 3, self.height * 16, self.width * 16)
        return [i_s, input_shape[1], input_shape[2], input_shape[3], input_shape[4]]



    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

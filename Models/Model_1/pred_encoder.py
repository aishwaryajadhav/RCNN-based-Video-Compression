import numpy as np
import torch
from torch.autograd import Variable

from network import EncoderCell
from network import Binarizer
from network import DecoderCell

class EncoderImage:

    def __init__(self, model, iterations=8, cuda = False):
        self.model = model
        self.cuda = cuda
        self.iterations = iterations
        
        encoder = EncoderCell()
        binarizer = Binarizer()
        decoder = DecoderCell()

        encoder.eval()
        binarizer.eval()
        decoder.eval()

        encoder.load_state_dict(torch.load(model))
        binarizer.load_state_dict(
            torch.load(model.replace('encoder', 'binarizer')))
        decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder')))

        if self.cuda:
            encoder = encoder.cuda()
            binarizer = binarizer.cuda()
            decoder = decoder.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.binarizer = binarizer

        # self.compress_error = 0

    def encode(self, image, size, iterations):
        
        image = Variable(torch.from_numpy(image))
        batch_size, input_channels, height, width = image.size()
        # print(image.size())
        output = np.empty((batch_size, input_channels, height, width))

        for i in range(0, height, size[0]):
            for j in range(0, width, size[1]):
                img = image[:,:,i:i+size[0], j:j+size[1]]
                img = self.encode_blocks(img, iterations)
                output[:,:, i : i+size[0], j : j+size[1]] = img.data.cpu().numpy()

        return output


    def encode_blocks(self, image, iterations):
        print('In pytorch!')  

        batch_size, input_channels, height, width = image.size()
        
        encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                    Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))
        encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                    Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))
        encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                    Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

        decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                    Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))
        decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                    Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))
        decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                    Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))
        decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)),
                    Variable(torch.zeros(batch_size, 128, height // 2, width // 2)))

    
        if self.cuda:
            image = image.cuda()

            encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
            encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
            encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())


        res = image - 0.5
        orig_image = res

    
        for iters in range(iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            code = self.binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
                code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

            res = orig_image - output
            
            # c_error = res.data.abs().mean()
            # loss.append(c_error)

            # if type == 'sub' and c_error <= self.compress_error:
            #     break

        # if type == 'first':
        #     self.compress_error = c_error

        # print('Iterations: {} .  '.format(len(loss)))
        # print(('{:.4f} ' * len(loss) +
        #             '\n').format(* [l for l in loss]))

        del res
        del orig_image
        del image
        
        
        # print(output.size())

        # my_K.variable = tf.get_K.variable("my_K.variable", [1, 2, 3])

        return output + 0.5


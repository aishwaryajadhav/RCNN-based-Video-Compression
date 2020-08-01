import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import videodataset

## load networks on GPU
import network
import vcnetwork


def resume(epoch=None):

    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    vcencoder.load_state_dict(
        torch.load('checkpoint/vcencoder_{}_{:08d}.pth'.format(s, epoch)))
    vcbinarizer.load_state_dict(
        torch.load('checkpoint/vcbinarizer_{}_{:08d}.pth'.format(s, epoch)))
    vcdecoder.load_state_dict(
        torch.load('checkpoint/vcdecoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(vcencoder.state_dict(), 'checkpoint/vcencoder_{}_{:08d}.pth'.format(
        s, index))

    torch.save(vcbinarizer.state_dict(),
               'checkpoint/vcbinarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(vcdecoder.state_dict(), 'checkpoint/vcdecoder_{}_{:08d}.pth'.format(
        s, index))




#################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size', '-N', type=int, default=16, help='batch size')
    parser.add_argument(
        '--train', '-f', required=True, type=str, help='folder of training images')
    parser.add_argument(
        '--max-epochs', '-e', type=int, default=10, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    # parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
    parser.add_argument(
        '--iterations', type=int, default=8, help='unroll iterations')
    parser.add_argument(
        '--timedepth', '-td' , type=int, default=30, help='Number of frames/Time Depth')
    # parser.add_argument('--checkpoint', type=int, help='unroll iterations')
    args = parser.parse_args()


    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = videodataset.VideoDataset(rootDir=args.train, timeDepth= args.timedepth ,transform=train_transform)

    train_loader = data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print('total videos: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))


    encoder = network.EncoderCell().cuda()
    binarizer = network.Binarizer().cuda()
    decoder = network.DecoderCell().cuda()
    
    encoder.eval()
    binarizer.eval()
    decoder.eval()

    # a = 'epoch'
    # b = 4

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_epoch_00000004.pth'))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_epoch_00000004.pth'))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_epoch_00000004.pth'))



    vcencoder = vcnetwork.EncoderCell().cuda()
    vcbinarizer = vcnetwork.Binarizer().cuda()
    vcdecoder = vcnetwork.DecoderCell().cuda()
    


    solver = optim.Adam(
        [
            {
                'params': vcencoder.parameters()
            },
            {
                'params': vcbinarizer.parameters()
            },
            {
                'params': vcdecoder.parameters()
            },
        ],
        lr=args.lr)


    resume()

    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    last_epoch = 0
    # resume()
    scheduler.last_epoch = last_epoch - 1

    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        
        scheduler.step()

        for batch, data in enumerate(train_loader):
            batch_t0 = time.time()
            tlf = args.timedepth - 1
            ## init lstm state
            
            # def commented_code:
                # encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                #             Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
                # encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                #             Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
                # encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                #             Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

                # decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                #             Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
                # decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                #             Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
                # decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                #             Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
                # decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                #             Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

            patches = Variable(data.cuda())

            # def commented_code:
                # res = patches[:,0,:,:,:]           
                # res = res - 0.5
                # orig_image = res

                # for _ in range(16):
                #     encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                #         res, encoder_h_1, encoder_h_2, encoder_h_3)

                #     codes = binarizer(encoded)

                #     output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                #         codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                #     res = orig_image - output

                
                # del orig_image
                # del res

            solver.zero_grad()

            prev = patches[:,0,:,:,:]  
            all_losses = []
            
            for i in range(tlf):
                # losses = []

                encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                            Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
                encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                            Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
                encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                            Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

                decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                            Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
                decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                            Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
                decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                            Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
                decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                            Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))


                res = patches[:,i+1,:,:,:]
                res = res - prev 
                # orig_image = res  
                

                # for _ in range(args.iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = vcencoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = vcbinarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = vcdecoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                # res = orig_image - output
                res = res - output
                all_losses.append(res.abs().mean())

                # print(('{:.8f} ' * args.iterations +
                # '\n').format(* [l.data.item() for l in losses]))

            
                # all_losses.append(sum(losses) / args.iterations)
                
                del res
                # del orig_image

                prev = patches[:,i+1,:,:,:]

            del prev

            loss = sum(all_losses) / tlf
            loss.backward()

            torch.nn.utils.clip_grad_norm_(vcencoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(vcbinarizer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(vcdecoder.parameters(), 1.0)

            solver.step()
            
            del encoder_h_1
            del encoder_h_2
            del encoder_h_3
            del decoder_h_1
            del decoder_h_2
            del decoder_h_3
            del decoder_h_4

            batch_t1 = time.time()



            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f};  Batch: {:.4f} sec'.
                format(epoch, batch + 1,
                    len(train_loader), loss.data.item(),  batch_t1 -
                    batch_t0))
            print(('{:.4f} ' * tlf +
                '\n').format(* [l.data.item() for l in all_losses]))
            # print('***********************')

            index = (epoch - 1) * len(train_loader) + batch

            ## save checkpoint every 50 training steps
            if index !=0 and index % 50 == 0:
                save(0, False)

            
        save(epoch)

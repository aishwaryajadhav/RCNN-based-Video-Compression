

class EncoderImage:

  def __init__(self, model, type_c=0, iterations=8, cuda = False):
    self.model = model
    self.cuda = cuda
    self.iterations = iterations
    self.type = type_c

    if self.type:
        encoder = EncoderCell()
        binarizer = Binarizer()
        decoder = DecoderCell()
    else:
        encoder = VEncoderCell()
        binarizer = VBinarizer()
        decoder = VDecoderCell()

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

  
  def encode_blocks(self, image, encoder_h_1=0, encoder_h_2=0, encoder_h_3=0, decoder_h_1=0, decoder_h_2=0, decoder_h_3=0, decoder_h_4=0, prev_deco = 0, count = 0):

    batch_size, input_channels, height, width = image.size()

    
    if encoder_h_1 == 0:
        prev_deco = Variable(torch.zeros(batch_size, input_channels, height, width))
      
        encoder_h_1 = (Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 256, height // 4, width // 4),
                        volatile=True))
        encoder_h_2 = (Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 512, height // 8, width // 8),
                        volatile=True))
        encoder_h_3 = (Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 512, height // 16, width // 16),
                        volatile=True))

        decoder_h_1 = (Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 512, height // 16, width // 16),
                        volatile=True))
        decoder_h_2 = (Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 512, height // 8, width // 8),
                        volatile=True))
        decoder_h_3 = (Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 256, height // 4, width // 4),
                        volatile=True))
        decoder_h_4 = (Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 128, height // 2, width // 2),
                        volatile=True))


    # print(encoder_h_1)

    
    assert height % 32 == 0 and width % 32 == 0

    image = Variable(image, volatile=True)
    
    if self.cuda:
        image = image.cuda()

        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())


    codes = []
    res = image - 0.5
    orig_image = res

    for iters in range(self.iterations):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        code = self.binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = orig_image - output

        codes.append(code.data.cpu().numpy())

  #         print(code.data.cpu().numpy().shape)
        print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))
        
      
    output_img = output + 0.5 + prev_deco
    output_img_store = np.squeeze(output_img.detach().numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images/d_'+str(count)+'.jpeg',output_img_store)
            

    del res
    del orig_image
    del image
    
    
    # return codes    # shape = 16 * 1 * 32 * 22 * 40 (for a 352*640 image)
    codes = (np.stack(codes).astype(np.int8) + 1) // 2
    # print(codes.shape)
    return codes, output_img, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4


  def encode(self, input_image, output_file, size=128):
    
    image = np.array(Image.open(input_image).convert('RGB'), dtype=np.float32)
    
    image = torch.from_numpy(
      np.expand_dims(
          np.transpose(image / 255.0, (2, 0, 1)), 0))

    batch_size, input_channels, height, width = image.size()
      
    ht = (height // size) * size
    wd = (width // size) * size
    
    image = image[:,:, :ht, :wd]
    
    codes = []
    
    for i in range(0, ht, size):
      codes_row = []
      for j in range(0, wd, size):
        img = image[:,:,i:i+size, j:j+size]
        c, _, _, _, _, _, _, _ = self.encode_blocks(img)
        codes_row.append(c)
        del img
      codes_row = np.stack(codes_row, 0)
      codes.append(codes_row)

    codes = np.stack(codes, 0)
    print(codes.shape)

    export = np.packbits(codes.reshape(-1))

    np.savez_compressed(output_file, shape=codes.shape, codes=export)
  

class EncoderVideo:

    def __init__(self, model1, model2, iterations1=16, iterations2=8, cuda = False, size = 128):
        self.model1 = model1
        self.model2 = model2
        self.cuda = cuda
        self.size = size
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.encoder_1 = EncoderImage(model1, type_c = 1, iterations = iterations1, cuda=cuda)
        self.encoder_2 = EncoderImage(model2, type_c = 0, iterations = iterations2, cuda=cuda) 

        self.k_codes = 0
        self.s_codes = 0
        
          
        self.count_e_img = 0
      
    def encode(self, input_video, output_file):
        vidcap = VideoFileClip(input_video)
        
        ht = (vidcap.h // self.size) * self.size
        wd = (vidcap.w // self.size) * self.size

        print(ht)
        print(wd)
    
        k_codes = []
        s_codes = []
        
        for i in range(0, ht, self.size):
            kc_row = []
            sc_row = []
            for j in range(0, wd, self.size):
                new_clip = vidcap.crop( x1 = j , y1 = i , width = self.size, height = self.size)
                kc, sc = self.encode_blocks(new_clip)
                kc_row.append(kc)
                sc_row.append(sc)
                del new_clip
            kc_row = np.stack(kc_row, 0)
            sc_row = np.stack(sc_row, 0)
            k_codes.append(kc_row)
            s_codes.append(sc_row)
        
        del vidcap
        k_codes = np.stack(k_codes, 0)  #(i * j * 6 * 16 * 1 * 32 * 2 * 2)
        s_codes = np.stack(s_codes, 0)  #(i * j * 60 * 8 * 1 * 16 * 2 * 2)

        self.k_codes = k_codes
        self.s_codes = s_codes
    
        print(key_codes.shape)
        print(sec_codes.shape)

        f_shape = key_codes.shape + sec_codes.shape
        print(f_shape)

        codes = np.concatenate((key_codes.reshape(-1), sec_codes.reshape(-1)), axis = 1)

        export = np.packbits(codes.reshape(-1))
        print(codes.reshape(-1).shape)

        np.savez_compressed(output_file, shape=f_shape, codes=export)

        
    def encode_blocks(self, clip):
        c = 10
        key_codes = []
        sec_codes = []
        mq = 0
        prev = 0
        for image in clip.iter_frames(dtype=np.float32):
            if(mq == 30):
                break
            mq = mq + 1
            
            cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images/'+str(self.count_e_img)+'.jpeg',image)
            self.count_e_img = self.count_e_img + 1
#             cv2_imshow(image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

            image = torch.from_numpy(
            np.expand_dims(
                np.transpose(image / 255.0, (2, 0, 1)), 0))
            
            if c == 10:
                c = 0
                codes, decoded_image, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.encoder_1.encode_blocks(image, count = self.count_e_img -1)
                key_codes.append(codes)  # (16 * 1 * 32 * 2 * 2)
            else:
                codes, decoded_image, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.encoder_2.encode_blocks(image - prev, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, prev_deco=decoded_image, count = self.count_e_img -1)
                c = c + 1
                sec_codes.append(codes)  # (8 * 1 * 16 * 2 * 2)

            prev = image
            del image

        key_codes = np.stack(key_codes, 0)  # (6 * 16 * 1 * 32 * 2 * 2)
        sec_codes = np.stack(sec_codes, 0)  # (60 * 8 * 1 * 16 * 2 * 2)

        return key_codes, sec_codes
    

class DecoderImage:
    
    def __init__(self, model, type_c=0, iterations=8, cuda = False, size = 128):
        self.model = model
        self.cuda =cuda
        self.iterations = iterations
        self.type = type_c
        self.size = size
        if self.type:
            decoder = DecoderCell()
        else:
            decoder = VDecoderCell()

        decoder.eval()

        decoder.load_state_dict(torch.load(model))

        if self.cuda:
            decoder = decoder.cuda()
        
        self.decoder = decoder


        
    def decode(self, input_code, output_loc, save_format):
        content = np.load(input_code)
        codes = np.unpackbits(content['codes'])
        codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

        codes = torch.from_numpy(codes)
        blocks_ht, blocks_wd, iters, batch_size, channels, height, width = codes.size()

        height_size = int(height / 2 * 32)
        width_size = int(width / 2 * 32)

        height = height_size * blocks_ht
        width = width_size * blocks_wd

        image = np.empty((height, width, 3))


        for i in range(0, blocks_ht):
            for j in range(0, blocks_wd):

                img = self.decode_blocks(codes[i,j,:,:,:,:,:])
                image[i*height_size : i*height_size+height_size, j*width_size : j*width_size+width_size, :] = img
    
        image = Image.fromarray(image.astype(np.uint8))      

        op = '{}_{}.{}'.format(os.path.splitext(input_code)[0] ,str(iters), save_format)
        op = os.path.join(output_loc, op)
        
        image.save(op)



    def decode_blocks(self, codes, decoder_h_1=0, decoder_h_2=0, decoder_h_3=0, decoder_h_4=0, count = 0, prev_decoded = 0):

        it, batch_size, channels, height, width = codes.size()
        
        height = height * 16
        width = width * 16

        if decoder_h_1 == 0:
            prev_decoded = torch.zeros((1, 3, self.size, self.size))

            decoder_h_1 = (Variable(
                torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                    Variable(
                        torch.zeros(batch_size, 512, height // 16, width // 16),
                        volatile=True))
            decoder_h_2 = (Variable(
                torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                        Variable(
                            torch.zeros(batch_size, 512, height // 8, width // 8),
                            volatile=True))
            decoder_h_3 = (Variable(
                torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                        Variable(
                            torch.zeros(batch_size, 256, height // 4, width // 4),
                            volatile=True))
            decoder_h_4 = (Variable(
                torch.zeros(batch_size, 128, height // 2, width // 2), volatile=True),
                        Variable(
                            torch.zeros(batch_size, 128, height // 2, width // 2),
                            volatile=True))


        codes = Variable(codes, volatile=True)

        if self.cuda:
            codes = codes.cuda()

            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

            
#         for iters in range(min(self.iterations, it)):
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
            codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        
        img =  output.data.cpu() + 0.5 + prev_decoded

        image = np.squeeze(img.numpy().clip(0, 1) * 255.0).transpose(1, 2, 0)

        cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images/'+str(count)+'.jpeg',image)
#         count_d_img = count_d_img + 1
#           
#         cv2_imshow(img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        

        return image, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, img


class DecoderVideo:

    def __init__(self, model1, model2, iterations1=16, iterations2=8, cuda = False, size = 128):
        self.model1 = model1
        self.model2 = model2
        self.cuda = cuda
        self.size = size
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.decoder_1 = DecoderImage(model1, type_c = 1, iterations = iterations1, cuda=cuda)
        self.decoder_2 = DecoderImage(model2, type_c = 0, iterations = iterations2, cuda=cuda) 


        self.k_codes = 0
        self.s_codes = 0
        
          
        self.count_d_img = 0
        
    def decode(self, input_code, output_loc, file_name):
        content = np.load(input_code)
        
        f_shape = content['shape']
        l = len(f_shape)

        key_shape = f_shape[:l//2]
        sec_shape = f_shape[l//2:]
      
        print(key_shape)
        print(sec_shape)

        codes = np.unpackbits(content['codes'])
        
        n = np.prod(key_shape)
        key_codes = codes[:n]
        sec_codes = codes[n:]

        self.k_codes = np.reshape(key_codes, key_shape)
        self.s_codes = np.reshape(sec_codes, sec_shape)

        key_codes = np.reshape(key_codes, key_shape).astype(np.float32) * 2 - 1
        sec_codes = np.reshape(sec_codes, sec_shape).astype(np.float32) * 2 - 1

        key_codes = torch.from_numpy(key_codes)
        sec_codes = torch.from_numpy(sec_codes)

        length = key_shape[2]+sec_shape[2]
        print(length)

        video = np.empty((length, self.size*key_shape[0], self.size*key_shape[1], 3))

        for i in range(key_shape[0]):
            x = i*self.size
            for j in range(key_shape[1]):
                l = 0
                y = j*self.size
                for k in range(key_shape[2]):
                    video[l, x : x+self.size, y : y+self.size, :], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, img = self.decoder_1.decode_block(key_codes[i,j,k,:,:,:,:,:], count = self.count_d_img)
                    l = l+1
                    c = 0
                    self.count_d_img = self.count_d_img + 1
                    while c < 10 and l<length:
                        video[l, x : x+self.size, y : y+self.size, :], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, img = self.decoder_2.decode_block(sec_codes[i,j,k*10+c,:,:,:,:,:],
                        decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,  count = self.count_d_img, prev_decoded = img)
                        self.count_d_img = self.count_d_img + 1
                        c = c+1
                        l = l+1
                    
        
        video = video.astype(np.uint8)
        skvideo.io.vwrite(os.path.join(output_loc,file_name), video)



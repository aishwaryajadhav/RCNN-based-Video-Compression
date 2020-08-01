class DecoderVideo:

    def __init__(self, model1, iterations1=16, iterations2=8, cuda = False, size = 128):
        self.model1 = model1
#         self.model2 = model2
        self.cuda = cuda
        self.size = size
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.decoder_1 = DecoderImage(model1, iterations1 = iterations1, iterations2 = iterations2, cuda=cuda)
#         self.decoder_2 = DecoderImage(model2, iterations = iterations2, cuda=cuda)         
          
        
    def decode(self, input_code, output_loc, file_name):
        content = np.load(input_code)
        
        f_shape = content['shape']
        l = len(f_shape)

        key_shape = f_shape[:l//2]
        sec_shape = f_shape[l//2:]
      
  
        codes = np.unpackbits(content['codes'])
        
        n = np.prod(key_shape)
        key_codes = codes[:n]
        sec_codes = codes[n:]


        key_codes = np.reshape(key_codes, key_shape).astype(np.float32) * 2 - 1
        sec_codes = np.reshape(sec_codes, sec_shape).astype(np.float32) * 2 - 1

        key_codes = torch.from_numpy(key_codes)
        sec_codes = torch.from_numpy(sec_codes)

        length = key_shape[2]+sec_shape[2]
#         print(length)

        video = np.empty((length, self.size*key_shape[0], self.size*key_shape[1], 3))

        for i in range(key_shape[0]):
            x = i*self.size
            for j in range(key_shape[1]):
                l = 0
                y = j*self.size
                for k in range(key_shape[2]):
                    video[l, x : x+self.size, y : y+self.size, :], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder_1.decode_block(key_codes[i,j,k,:,:,:,:,:], type = 0)
                    l = l+1
                    c = 0
                    while c < 10 and l<length:
                        video[l, x : x+self.size, y : y+self.size, :], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder_1.decode_block(sec_codes[i,j,k*10+c,:,:,:,:,:], type = 1, decoder_h_1 = decoder_h_1, decoder_h_2= decoder_h_2, decoder_h_3 = decoder_h_3, decoder_h_4 = decoder_h_4)
                        c = c+1
                        l = l+1
                    
        
        video = video.astype(np.uint8)
        skvideo.io.vwrite(os.path.join(output_loc,file_name), video)



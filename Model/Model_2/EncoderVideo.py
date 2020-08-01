class EncoderVideo:

    def __init__(self, model1, iterations1=16, iterations2=8, cuda = False, size = 128):
        self.model1 = model1
#         self.model2 = model2
        self.cuda = cuda
        self.size = size
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.encoder_1 = EncoderImage(model1, iterations1 = iterations1, iterations2= iterations2, cuda=cuda)
#         self.encoder_2 = EncoderImage(model2, iterations = iterations2, cuda=cuda) 
  
        self.count_e_img = 0
    
      
    def encode(self, input_video, output_file):
        vidcap = VideoFileClip(input_video)
        
        ht = (vidcap.h // self.size) * self.size
        wd = (vidcap.w // self.size) * self.size
    
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


        f_shape = k_codes.shape + s_codes.shape

        codes = np.concatenate((k_codes.reshape(-1), s_codes.reshape(-1)))

        export = np.packbits(codes.reshape(-1))

        np.savez_compressed(output_file, shape=f_shape, codes=export)

        
    def encode_blocks(self, clip):
        c = 10
        key_codes = []
        sec_codes = []
        mq = 0
        prev = 0.0
        for image in clip.iter_frames(dtype=np.float32):
            if(mq == 20):
                break
            mq = mq + 1
            
#             if self.count_e_img >= 180 and self.count_e_img<= 190:    
#             cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images3/'+str(self.count_e_img)+'.jpeg',image)

#             diff = img - prev
#             cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images3/diff_orig_'+str(self.count_e_img)+'.jpeg', diff)

            self.count_e_img = self.count_e_img + 1
            
            
            image = torch.from_numpy(
            np.expand_dims(
                np.transpose(image / 255.0, (2, 0, 1)), 0))
            
#             diff = torch.from_numpy(
#             np.expand_dims(
#                 np.transpose(diff / 255.0, (2, 0, 1)), 0))
            
            
            if c == 10:
                c = 0
                codes, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.encoder_1.encode_blocks(image, type = 0, count = self.count_e_img-1)
                key_codes.append(codes)  # (16 * 1 * 32 * 2 * 2)
            else:
                codes, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.encoder_1.encode_blocks(image, type = 1, encoder_h_1 = encoder_h_1, encoder_h_2 = encoder_h_2, encoder_h_3= encoder_h_3, decoder_h_1 = decoder_h_1, decoder_h_2= decoder_h_2, decoder_h_3 = decoder_h_3, decoder_h_4 = decoder_h_4, count = self.count_e_img-1)
                c = c + 1
                sec_codes.append(codes)  # (8 * 1 * 16 * 2 * 2)

            

        key_codes = np.stack(key_codes, 0)  # (6 * 16 * 1 * 32 * 2 * 2)
        sec_codes = np.stack(sec_codes, 0)  # (60 * 8 * 1 * 16 * 2 * 2)

        return key_codes, sec_codes
    

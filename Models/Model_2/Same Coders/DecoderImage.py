class DecoderImage:
    
    def __init__(self, model,iterations1 = 16, iterations2=8, cuda = False, size = 128):
        self.model = model
        self.cuda =cuda
        self.iterations1 = iterations1
        self.iterations2 = iterations2
        self.size = size
        decoder = DecoderCell()
        
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



    def decode_block(self, codes, decoder_h_1=0, decoder_h_2=0, decoder_h_3=0, decoder_h_4=0, type = 0):

        it, batch_size, channels, height, width = codes.size()
        
        height = height * 16
        width = width * 16

        it_s = self.iterations2
        if type == 0:
            it_s = self.iterations1
#             prev_decoded = torch.zeros((1, 3, self.size, self.size))

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

            
        for iters in range(it_s):
          output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
              codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        
        img =  output.data.cpu() + 0.5                   # decoded image in a raw form

        image = np.squeeze(img.numpy().clip(0, 1) * 255.0).transpose(1, 2, 0)     # decoded image in a display ready form

        return image, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4

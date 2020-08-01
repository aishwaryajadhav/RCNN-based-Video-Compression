class EncoderImage:

  def __init__(self, model, iterations1 = 16, iterations2=8, cuda = False):
    self.model = model
    self.cuda = cuda
    self.iterations1 = iterations1
    self.iterations2 = iterations2
   
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

  
  def encode_blocks(self, image, type = 0, encoder_h_1=0, encoder_h_2=0, encoder_h_3=0, decoder_h_1=0, decoder_h_2=0, decoder_h_3=0, decoder_h_4=0, count = 0):

    batch_size, input_channels, height, width = image.size()

    it_s = self.iterations2
    
    if type == 0:
        it_s = self.iterations1
      
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

    
    if(height % 32 != 0 or width % 32 != 0):
      print('{} and {} '.format(height, width))
      image = torch.zeros((1,3,128,128))

    image = Variable(image, volatile=True)
    
    if self.cuda:
      image = image.cuda()
#         prev_deco = prev_deco.cuda()

      encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
      encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
      encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

      decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
      decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
      decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
      decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())


    
    codes = []
    image = image -0.5
    res = image
    
    for iters in range(it_s):
      encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
          res, encoder_h_1, encoder_h_2, encoder_h_3)

      code = self.binarizer(encoded)

      output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
          code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

      res = image - output

      codes.append(code.data.cpu().numpy())

      print('Iters: {} Loss: {:.06f}'.format(iters, res.data.abs().mean()))


#     output_img = output.cpu() + prev_decoded + 0.5
#     output = output.data.cpu() + 0.5

#     if self.count_e_img >= 180 and self.count_e_img<= 190:          
#     output_img_store = np.squeeze(output.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)

      
      #     diff_img_store = np.squeeze(op.detach().numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    
#     cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images3/decoded_'+str(count)+'.jpeg',output_img_store)
#     cv2.imwrite('/content/drive/My Drive/Google collab/Compression/output/Images3/diff_'+str(count)+'.jpeg',diff_img_store)

    codes = (np.stack(codes).astype(np.int8) + 1) // 2
    
    return codes, encoder_h_1, encoder_h_2, encoder_h_3, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4


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
  


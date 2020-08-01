
import argparse
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from PIL import Image

class Metric:

    def _FSpecialGauss( size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1
        x, y = np.mgrid[offset + start:stop, offset + start:stop]
        assert len(x) == size
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()


    def _SSIMForMultiScale( img1,img2,max_val=255,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03):
        
        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape,img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        _, height, width, _ = img1.shape

        # Filter size can't be larger than height or width of images.
        size = min(filter_size, height, width)

        # Scale down sigma if a smaller filter size is used.
        sigma = size * filter_sigma / filter_size if filter_size else 0

        if filter_size:
            window = np.reshape(Metric._FSpecialGauss(size, sigma), (1, size, size, 1))
            mu1 = signal.fftconvolve(img1, window, mode='valid')
            mu2 = signal.fftconvolve(img2, window, mode='valid')
            sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
            sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
            sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
        else:
            # Empty blur kernel so no need to convolve.
            mu1, mu2 = img1, img2
            sigma11 = img1 * img1
            sigma22 = img2 * img2
            sigma12 = img1 * img2

        mu11 = mu1 * mu1
        mu22 = mu2 * mu2
        mu12 = mu1 * mu2
        sigma11 -= mu11
        sigma22 -= mu22
        sigma12 -= mu12

        # Calculate intermediate values used by both ssim and cs_map.
        c1 = (k1 * max_val)**2
        c2 = (k2 * max_val)**2
        v1 = 2.0 * sigma12 + c2
        v2 = sigma11 + sigma22 + c2
        ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
        cs = np.mean(v1 / v2)
        return ssim, cs


    def MultiScaleSSIM( img1,img2,max_val=255,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,weights=None):
      
        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape,img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',img1.ndim)

      # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
        weights = np.array(weights if weights else
                            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size
        downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
        im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
        mssim = np.array([])
        mcs = np.array([])
        for _ in range(levels):
            ssim, cs = Metric._SSIMForMultiScale(im1,im2,max_val=max_val,filter_size=filter_size,filter_sigma=filter_sigma,k1=k1,k2=k2)
            mssim = np.append(mssim, ssim)
            mcs = np.append(mcs, cs)
            filtered = [
                convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]
            ]
            im1, im2 = [x[:, ::2, ::2, :] for x in filtered]

        return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
                (mssim[levels - 1]**weights[levels - 1]))


    def msssim( original, compared):
  
      # compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)
        ht, wd, _ = compared.shape
    
        # if isinstance(original, str):
        # original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
        original = original[:ht, :wd, :]

    
        original = original[None, ...] if original.ndim == 3 else original
        compared = compared[None, ...] if compared.ndim == 3 else compared

        return Metric.MultiScaleSSIM(original, compared, max_val=255)


    def psnr(original, compared):
      # if isinstance(compared, str):
        # compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)
        ht, wd, _ = compared.shape
        print(compared.shape)
        # if isinstance(original, str):
        # original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
        original = original[:ht, :wd, :]
    
        mse = np.mean(np.square(original - compared))
        psnr = np.clip(
            np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
        return psnr


    def calc_metric_image(original_file, compared_file):
      # if args.metric != 'psnr':
        original_image = np.array(Image.open(original_file).convert('RGB'), dtype=np.float32)
        compared_image = np.array(Image.open(compared_file).convert('RGB'), dtype=np.float32)

        return Metric.calc_metric(original_image, compared_image)
      

    def calc_metric(original_image, compared_image):
        # if args.metric != 'psnr':
        ssim = Metric.msssim(original_image, compared_image)
        print("\n")
        # if args.metric != 'ssim':
        psnr = Metric.psnr(original_image, compared_image)
        return (ssim, psnr)

    def cal_metric_video(video_file_orig, video_file_compared, nt):
        # nt = nt - 1

        original = cv2.VideoCapture(video_file_orig)
        compared = cv2.VideoCapture(video_file_compared)

        width = int(compared.get(3))
        height = int(compared.get(4))            
        
        success1, frame1 = original.read()
        # success2, frame2 = compared.read()
        
        success1, frame1 = original.read()
        success2, frame2 = compared.read()
        
        psnr_sum = 0
        mssim_sum = 0
        c = 0
        mse_sum = 0
        
        while success1 and success2 and c < nt:
            
            mse_sum = mse_sum + ((frame1 - frame2)**2)
            (s, p) = Metric.calc_metric(frame1, frame2)
            psnr_sum = psnr_sum + p
            mssim_sum = mssim_sum + s
            c = c + 1
            success1, frame1 = original.read()
            success2, frame2 = compared.read()

        psnr = psnr_sum / nt
        mssim = mssim_sum / nt
        mse_sum = np.mean(mse_sum) / nt
        return (mssim, psnr, mse_sum)

    
    def calc_metric_arrays(orig_array, recon_array, nt): 
        psnr_sum = 0
        mssim_sum = 0
        mse = 0
        for i in range(1, nt):
            (s, p) = Metric.calc_metric(orig_array[i], recon_array[i])
            mse = mse + ((orig_array[i] - recon_array[i])**2)
            psnr_sum = psnr_sum + p
            mssim_sum = mssim_sum + s

        psnr = psnr_sum / (nt-1)
        mssim = mssim_sum /( nt-1)
        mse = np.mean(mse) / (nt-1)
        return (mssim, psnr, mse)




import numpy as numpy
import os
from moviepy.editor import *
from metric_image_video import Metric


# kbps = ['900k', '1800k', '2700k', '3600k', '4500k', '5600k', '6300k', '7200k', '8100k', '9000k', '9900k', '10800k']

parent_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\show\\t'

output_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\show\\result'

for f in os.listdir(parent_dir):
    file_name = os.path.join(parent_dir, f)
    video = VideoFileClip(file_name)
    # for i, kb in enumerate(kbps):
    video.write_videofile(os.path.join(output_dir, f+'264'+'.mp4'), codec= 'libx264')


# output_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\youtube\\mpeg4_youtube'

for f in os.listdir(parent_dir):
    file_name = os.path.join(parent_dir, f)
    video = VideoFileClip(file_name)
    # for i, kb in enumerate(kbps):
    video.write_videofile(os.path.join(output_dir, f+'mpeg'+'.mp4'), codec= 'mpeg4')


# BASE_DIR = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\t5'
# RESULTS_SAVE_DIR=os.path.join(BASE_DIR,'Results')

# original_dir = os.path.join(BASE_DIR, 't')
# test_dir = os.path.join(BASE_DIR, 'test')
# h264_dir = os.path.join(BASE_DIR, 'h264')
# mpeg4_dir = os.path.join(BASE_DIR, 'mpeg4')
# pe_dir = os.path.join(BASE_DIR, 'predencoder')

# h264_mssim_file = 'h264_mssim_file.csv'
# h264_psnr_file = 'h264_psnr_file.csv'
# h264_mse_file = 'h264_mse_file.csv'

# mpeg4_mssim_file =  'mpeg4_mssim_file.csv'
# mpeg4_psnr_file =  'mpeg4_psnr_file.csv'
# mpeg4_mse_file = 'mpeg4_mse_file.csv'

# pe_mssim_file =  'pe_mssim_file.csv'
# pe_psnr_file =  'pe_psnr_file.csv'
# pe_mse_file =  'pe_mse_file.csv'

# hm = open(os.path.join(RESULTS_SAVE_DIR , h264_mssim_file), 'w')
# hp = open(os.path.join(RESULTS_SAVE_DIR , h264_psnr_file), 'w')
# he = open(os.path.join(RESULTS_SAVE_DIR , h264_mse_file), 'w')

# mm = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_mssim_file), 'w')
# mp = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_psnr_file), 'w')
# me = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_mse_file), 'w')

# pm = open(os.path.join(RESULTS_SAVE_DIR , pe_mssim_file), 'w')
# pp = open(os.path.join(RESULTS_SAVE_DIR , pe_psnr_file), 'w')
# pe = open(os.path.join(RESULTS_SAVE_DIR , pe_mse_file), 'w')


# hm.write("")
# hp.write("")
# he.write("")

# mm.write("")
# mp.write("")
# me.write("")

# pm.write("")
# pp.write("")
# pe.write("")


# for f in os.listdir(test_dir):
#     for i in range(12):
        # new_file = f + '_'+str(i)+'.mp4'
        # (mssim, psnr, mse) = Metric.cal_metric_video(os.path.join(original_dir, f), os.path.join(h264_dir, new_file), 10)
        # (mssim1, psnr1, mse1) = Metric.cal_metric_video(os.path.join(original_dir, f), os.path.join(mpeg4_dir, new_file), 10)
        
        # test_file = f+'.avi'
        # new_file = f + '_'+str(i)+'.avi'
        # (mssim2, psnr2, mse2) = Metric.cal_metric_video(os.path.join(test_dir, f), os.path.join(pe_dir, new_file), 9)
        
        # hm.write("%f, " % mssim)
        # hp.write("%f, " % psnr)
        # he.write("%f, " % mse)
        
        # mm.write("%f, " % mssim1)
        # mp.write("%f, " % psnr1)
        # me.write("%f, " % mse1)

        # pm.write("%f, " % mssim2)
        # pp.write("%f, " % psnr2)
        # pe.write("%f, " % mse2)

    # hm.write("\n")
    # hp.write("\n")
    # he.write("\n")

    # mm.write("\n")
    # mp.write("\n")
    # me.write("\n")

    # pm.write("\n")
    # pp.write("\n")
    # pe.write("\n")


# hm.close()
# he.close()
# hp.close()

# mm.close()
# mp.close()
# me.close()

# pm.close()
# pp.close()
# pe.close()



# import numpy as numpy
# import os
# from moviepy.editor import *
# from metric_image_video import Metric


# # kbps = ['900k', '1800k', '2700k', '3600k', '4500k', '5600k', '6300k', '7200k', '8100k', '9000k', '9900k', '10800k']

# # parent_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\youtube\\avifiles'

# # output_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\youtube\\h264_youtube'

# # for f in os.listdir(parent_dir):
# #     file_name = os.path.join(parent_dir, f)
# #     video = VideoFileClip(file_name)
# #     for i, kb in enumerate(kbps):
# #         video.write_videofile(os.path.join(output_dir, f+'_'+str(i)+'.mp4'), codec= 'libx264', bitrate=kb)


# # output_dir = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\youtube\\mpeg4_youtube'

# # for f in os.listdir(parent_dir):
# #     file_name = os.path.join(parent_dir, f)
# #     video = VideoFileClip(file_name)
# #     for i, kb in enumerate(kbps):
# #         video.write_videofile(os.path.join(output_dir, f+'_'+str(i)+'.mp4'), codec= 'mpeg4', bitrate=kb)


# BASE_DIR = 'D:\\meais\\Documents\\Research\\meais-sf\\youtube-8m-videos-frames-master\\t5'
# RESULTS_SAVE_DIR=os.path.join(BASE_DIR,'Results')

# original_dir = os.path.join(BASE_DIR, 't')
# test_dir = os.path.join(BASE_DIR, 'test')
# h264_dir = os.path.join(BASE_DIR, 'h264')
# mpeg4_dir = os.path.join(BASE_DIR, 'mpeg4')
# pe_dir = os.path.join(BASE_DIR, 'predencoder')

# h264_mssim_file = 'h264_mssim_file.csv'
# h264_psnr_file = 'h264_psnr_file.csv'
# h264_mse_file = 'h264_mse_file.csv'

# mpeg4_mssim_file =  'mpeg4_mssim_file.csv'
# mpeg4_psnr_file =  'mpeg4_psnr_file.csv'
# mpeg4_mse_file = 'mpeg4_mse_file.csv'

# # pe_mssim_file =  'pe_mssim_file.csv'
# # pe_psnr_file =  'pe_psnr_file.csv'
# # pe_mse_file =  'pe_mse_file.csv'

# hm = open(os.path.join(RESULTS_SAVE_DIR , h264_mssim_file), 'w')
# hp = open(os.path.join(RESULTS_SAVE_DIR , h264_psnr_file), 'w')
# he = open(os.path.join(RESULTS_SAVE_DIR , h264_mse_file), 'w')

# mm = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_mssim_file), 'w')
# mp = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_psnr_file), 'w')
# me = open(os.path.join(RESULTS_SAVE_DIR , mpeg4_mse_file), 'w')

# # pm = open(os.path.join(RESULTS_SAVE_DIR , pe_mssim_file), 'w')
# # pp = open(os.path.join(RESULTS_SAVE_DIR , pe_psnr_file), 'w')
# # pe = open(os.path.join(RESULTS_SAVE_DIR , pe_mse_file), 'w')


# hm.write("")
# hp.write("")
# he.write("")

# mm.write("")
# mp.write("")
# me.write("")

# # pm.write("")
# # pp.write("")
# # pe.write("")


# for f in os.listdir(original_dir):
#     for i in range(12):
#         new_file = f + '_'+str(i)+'.mp4'
#         (mssim, psnr, mse) = Metric.cal_metric_video(os.path.join(original_dir, f), os.path.join(h264_dir, new_file), 10)
#         (mssim1, psnr1, mse1) = Metric.cal_metric_video(os.path.join(original_dir, f), os.path.join(mpeg4_dir, new_file), 10)
        
#         # new_file = f + '_'+str(i)+'.avi'
#         # (mssim2, psnr2, mse2) = Metric.cal_metric_video(os.path.join(test_dir, f), os.path.join(pe_dir, new_file), 10)
        
#         hm.write("%f, " % mssim)
#         hp.write("%f, " % psnr)
#         he.write("%f, " % mse)
        
#         mm.write("%f, " % mssim1)
#         mp.write("%f, " % psnr1)
#         me.write("%f, " % mse1)

#         # pm.write("%f, " % mssim2)
#         # pp.write("%f, " % psnr2)
#         # pe.write("%f, " % mse2)

#     hm.write("\n")
#     hp.write("\n")
#     he.write("\n")

#     mm.write("\n")
#     mp.write("\n")
#     me.write("\n")

#     # pm.write("\n")
#     # pp.write("\n")
#     # pe.write("\n")


# hm.close()
# he.close()
# hp.close()

# mm.close()
# mp.close()
# me.close()

# # pm.close()
# # pp.close()
# # pe.close()






# RCNN-based-Video-Compression

In recent years, neural network based image compression techniques have been able to outperform traditional codecs and have opened the gates for the development of learning-based video codecs. However, to take advantage of the high temporal correlation in videos, more sophisticated architectures need to be employed. This paper presents PredEncoder, a hybrid video compression framework based on the concept of predictive auto-encoding that models the temporal correlations between consecutive video frames using a prediction network which is then combined with a progressive encoder network to exploit the spatial redundancies. A variable-rate block encoding scheme has been proposed in the paper that leads to remarkably high quality to bit-rate ratios. By joint training and fine-tuning of this hybrid architecture, PredEncoder has been able to gain significant improvement over the MPEG-4 codec and has achieved bit-rate savings over the H.264 codec in the low to medium bit-rate range for HD videos and comparable results over most bit-rates for non-HD videos. This paper serves to demonstrate how neural architectures can be leveraged to perform at par with the highly optimized traditional methodologies in the video compression domain.

ICML Presentation: slideslive.com/38931623

ICML Poster: https://drive.google.com/file/d/1dRyxz_Kg88YIRTNqEYos_tCQcclWRJq2/view?usp=sharing

Full paper: https://ieeexplore.ieee.org/document/9104085

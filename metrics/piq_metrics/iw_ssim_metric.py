import torch
from piq import information_weighted_ssim
from pycvvdp.vq_metric import vq_metric

"""
IW-SSIM metric. Usage is same as the ColorVideoVDP metric (see examples).
"""

"""
To be able to use this metric, you will need to install piq: $ pip install piq
"""

class iw_ssim_metric(vq_metric):

    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.color_space = 'display_encoded_dmax'

    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        score = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)

            score += information_weighted_ssim(T, R, data_range=1., k2=0.4) / N_frames

        return score, None

    def short_name(self):
        return "IW-SSIM"

    @staticmethod
    def name():
        return "IW-SSIM"
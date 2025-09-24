import torch
from piq import mdsi
from pycvvdp.vq_metric import vq_metric

"""
MDSI metric. Usage is same as the ColorVideoVDP metric (see examples).
"""

"""
To be able to use this metric, you will need to install piq: $ pip install piq
"""

class mdsi_metric(vq_metric):

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

            score += mdsi(T, R, data_range=1.) / N_frames

        return score, None

    def short_name(self):
        return "MDSI"

    @staticmethod
    def name():
        return "MDSI"

    @staticmethod
    def is_lower_better():
        return True

    @staticmethod
    def predictions_range():
        return 0, 1
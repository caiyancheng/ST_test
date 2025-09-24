import torch
from pyiqa import create_metric
from pycvvdp.vq_metric import vq_metric

"""
DISTS metric.
"""

"""
To be able to use this metric, you will need to install pyiqa: $ pip install pyiqa
"""

class dists_metric(vq_metric):

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

        self.metric = create_metric('dists', device=self.device)

    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        score = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)

            score += self.metric(T, R) / N_frames

        return score, None

    def short_name(self):
        return "DISTS"

    @staticmethod
    def name():
        return "DISTS"
    
    @staticmethod
    def is_lower_better():
        return True

    @staticmethod
    def predictions_range():
        return 0, 1
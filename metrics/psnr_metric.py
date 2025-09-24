import torch
from pyiqa import create_metric
from pycvvdp.vq_metric import vq_metric
from pyst.utils import RGB2Y

"""
PSNR metric.
"""

class psnry_metric(vq_metric):

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

        mse = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.color_space).squeeze(2).clamp(min=0, max=1)

            """ The metric excepts a gray image """
            T = RGB2Y(T, self.display_photometry)
            R = RGB2Y(R, self.display_photometry)

            mse += torch.mean( (T - R)**2 )
        
        score = 20*torch.log10( 1.0 /torch.sqrt(mse/N_frames) ) 

        return score, None

    def short_name(self):
        return "PSNR-Y"

    @staticmethod
    def name():
        return "PSNR-Y"
    
    @staticmethod
    def is_lower_better():
        return False

    @staticmethod
    def predictions_range():
        return 0, 100

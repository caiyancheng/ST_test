import torch
from pyiqa import create_metric
from pycvvdp.vq_metric import vq_metric
from subprocess import Popen, PIPE
import tempfile
import scipy


class scielab_metric(vq_metric):

    def __init__(self, device=None) -> None:

        super().__init__()

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        worker_func = 'worker_SCIELAB()'
        cmd = f'matlab -nodisplay -nodesktop -nojvm -nosplash -sd metrics/matlab_metrics -r display(pwd);{worker_func};quit;'
        self.proc = Popen(cmd.split(), stdin=PIPE, stdout=PIPE)

        self.colorspace = "display_encoded_01"


        # Skip Matlab's copyright message
        while True:
            mstr = self.proc.stdout.readline()
            if mstr is None or len(mstr) == 0:
                raise RuntimeError("Failed to read from the worker")
            # print( mstr )
            if mstr == b"<start>\n":
                break

    def __del__(self):
        # Tell the matlab metric to finish
        self.proc.stdin.write(bytes('q\n', 'utf-8'))

    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        Q = 0.0
        N = 0.0
        # if N_frames>1: # Otherwise, we get stuck on the last freame for some reason
        #     N_frames -= 1

        for ff in range(N_frames):

            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)


            with tempfile.NamedTemporaryFile(mode='w+b', prefix='SCIELAB' + "-", suffix=".mat") as tf:

                ppd = self.display_geo.get_ppd()
                frames = {'T': torch.permute(T, (3, 4, 1, 0, 2)).squeeze().cpu().numpy(),
                          'R': torch.permute(R, (3, 4, 1, 0, 2)).squeeze().cpu().numpy(), 'ppd': ppd}
                scipy.io.savemat(tf.name, frames)

                self.proc.stdin.write(bytes(f'c {tf.name}\n', 'utf-8'))
                self.proc.stdin.flush()
                res = self.proc.stdout.readline()

                Q_frame = float(res)
                # print( f"[Python] Q = {Q_frame}", file=sys.stderr )
                if Q_frame is None:
                    raise RuntimeError("Could not get the metric result")

            Q += Q_frame
            N += 1

        return torch.as_tensor(Q / N), None

    def short_name(self):
        return 'sCIELab'

    def quality_unit(self):
        return None

    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None):
        self.display_photo = display_photometry
        self.display_geo = display_geometry

    def get_info_string(self):
        return None

    @staticmethod
    def name():
        return "sCIELab"

    @staticmethod
    def is_lower_better():
        return True

    @staticmethod
    def predictions_range():
        return 0, 10
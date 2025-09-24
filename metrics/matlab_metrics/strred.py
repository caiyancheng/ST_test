import torch
from pyiqa import create_metric
from pycvvdp.vq_metric import vq_metric
from subprocess import Popen, PIPE
import tempfile
import scipy
import numpy as np
from pyst.yuv_utils import float2fixed

class strred_metric(vq_metric):

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

        worker_func = 'worker_STRRED()'
        cmd = f'matlab -nodisplay -nodesktop -nojvm -nosplash -sd metrics/matlab_metrics -r display(pwd);{worker_func};quit;'
        self.proc = Popen(cmd.split(), stdin=PIPE, stdout=PIPE)

        self.colorspace = "display_encoded_01"

        self.col_mat = np.array([[0.2126, 0.7152, 0.0722], \
                                      [-0.114572, -0.385428, 0.5], \
                                      [0.5, -0.454153, -0.045847]], dtype=np.float32)

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

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''

    def predict_video_source(self, vid_source, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        #
        # We assume the pytorch default NCDHW layout

        h, w, N_frames = vid_source.get_video_size()

        Q = 0.0
        N = 0.0

        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            T_enc_np = T.squeeze().permute(1, 2, 0).cpu().numpy()
            R_enc_np = R.squeeze().permute(1, 2, 0).cpu().numpy()

            # Save the output as yuv file

            max_lum = 255
            offset = 16
            weight = 219

            YUV = (np.reshape(T_enc_np, (h*w, 3), order='F') @ self.col_mat.transpose()).reshape((T_enc_np.shape),order='F')
            Yt = (weight*YUV[:,:,0]+offset).clip(0,max_lum)

            YUV = (np.reshape(R_enc_np, (h * w, 3), order='F') @ self.col_mat.transpose()).reshape((R_enc_np.shape),order='F')
            Yr = (weight*YUV[:,:,0]+offset).clip(0,max_lum)

            # s_indexes = []
            # t_indexes = []

            if N_frames == 1:
                with tempfile.NamedTemporaryFile(mode='w+b', prefix='STRRED' + "-", suffix=".mat") as tf:
                    frames = {'Yr': Yr, 'Yt': Yt, 'Yr_prev': Yr, 'Yt_prev': Yt, 'type': 'image'}
                    scipy.io.savemat(tf.name, frames)

                    self.proc.stdin.write(bytes(f'c {tf.name}\n', 'utf-8'))
                    self.proc.stdin.flush()
                    res = self.proc.stdout.readline()

                    srred = float(res.decode().split(" ")[0])


                    Q = srred

                    if Q is None:
                        raise RuntimeError("Could not get the metric result")
                N = 1

            else:
                if ff > 0:
                    with tempfile.NamedTemporaryFile(mode='w+b', prefix='STRRED'+ "-", suffix=".mat") as tf:
                        frames = {'Yr': Yr, 'Yt': Yt, 'Yr_prev': Yr_prev, 'Yt_prev': Yt_prev, 'type': 'video'}
                        scipy.io.savemat(tf.name, frames)

                        self.proc.stdin.write(bytes(f'c {tf.name}\n', 'utf-8'))
                        self.proc.stdin.flush()
                        res = self.proc.stdout.readline()

                        srred = float(res.decode().split(" ")[0])
                        trred = float(res.decode().split(" ")[1])

                        Q_frame = srred * trred

                        if Q_frame is None:
                            raise RuntimeError("Could not get the metric result")

                        # if self.metric_name == 'SPEEDQA':
                        #     s_indexes.append(srred)
                        #     t_indexes.append(trred)

                    N += 1
                else:
                    Q_frame = 0

                Q += Q_frame
                Yr_prev = Yr
                Yt_prev = Yt

        # s_indexes.append(srred)
        # t_indexes.append(trred)

        return torch.as_tensor(Q / N), None
        # elif self.metric_name == 'SPEEDQA':
        #     return torch.as_tensor(np.mean(s_indexes) * np.mean(t_indexes)), None

    def set_display_model(self, display_photometry, display_geometry):
        self.max_L = display_photometry.get_peak_luminance()
        self.max_L = np.array(min(self.max_L, 300))

    def short_name(self):
        return 'ST-RRED'

    def quality_unit(self):
        return None

    def get_info_string(self):
        return None

    @staticmethod
    def name():
        return "ST-RRED"

    @staticmethod
    def is_lower_better():
        return False

    @staticmethod
    def predictions_range():
        return 0, 1000
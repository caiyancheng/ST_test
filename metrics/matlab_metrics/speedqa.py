import torch
from pyiqa import create_metric
from pycvvdp.vq_metric import vq_metric
from subprocess import Popen, PIPE
import tempfile
import scipy
import numpy as np
from pyst.yuv_utils import float2fixed
import os
import time

class speedqa_metric(vq_metric):

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

        # worker_func = 'worker_SPEEDQA()'
        # cmd = f'matlab -nodisplay -nodesktop -nojvm -nosplash -sd metrics/matlab_metrics -r display(pwd);{worker_func};quit;'
        self.matlab_cwd = os.path.abspath("metrics/matlab_metrics")
        self.log_file = os.path.join(self.matlab_cwd, "matlab_log.txt")
        self.cmd_file = os.path.join(self.matlab_cwd, "cmd.txt")

        # 清理残留
        os.makedirs(self.matlab_cwd, exist_ok=True)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists(self.cmd_file):
            os.remove(self.cmd_file)

        # 启动 MATLAB：GUI 子系统进程 -> 用 -logfile 让所有输出进日志文件
        cmd = [
            "matlab",
            "-nodesktop", "-nosplash",
            "-sd", self.matlab_cwd,
            "-logfile", self.log_file,
            "-r", "disp(pwd);worker_SPEEDQA();quit;"
        ]
        # 这里不需要管道；我们全靠日志通信
        self.proc = Popen(cmd)

        self.colorspace = "display_encoded_01"
        self.col_mat = np.array([
            [0.2126, 0.7152, 0.0722],
            [-0.114572, -0.385428, 0.5],
            [0.5, -0.454153, -0.045847]
        ], dtype=np.float32)

        # 记录当前日志读到的位置；等待 <start>
        self._log_pos = 0
        timeout = 60.0
        start_time = time.time()
        while True:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(self._log_pos)
                    lines = f.readlines()
                    self._log_pos = f.tell()
                if any("<start>" in ln for ln in lines):
                    break
            if time.time() - start_time > timeout:
                raise RuntimeError("MATLAB worker 启动超时 for SpeedQA（>60s），请检查是否正常运行。")
            time.sleep(0.1)

    # def __del__(self):
    #     # Tell the matlab metric to finish
    #     self.proc.stdin.write(bytes('q\n', 'utf-8'))
    def __del__(self):
        try:
            # 用命令文件请求退出
            if hasattr(self, "cmd_file") and self.cmd_file:
                tmp_cmd = self.cmd_file + ".tmp"
                with open(tmp_cmd, "w", encoding="utf-8") as f:
                    f.write("q\n")
                os.replace(tmp_cmd, self.cmd_file)
        except Exception:
            pass
        try:
            if hasattr(self, "proc") and self.proc:
                # 等一会儿让 MATLAB 自己退出
                self.proc.wait(timeout=5)
        except Exception:
            # 实在不行就不阻塞析构
            pass

    def _wait_for_result_in_log(self, timeout=10.0):
        """
        轮询 self.log_file，返回最近一次结果 (srred, trred)，并推进 self._log_pos。
        建议在 worker 的 fprintf 前加 '<res>' 前缀；若未加，也会兜底匹配“两浮点”一行。
        """
        import re
        start = time.time()
        float2 = re.compile(
            r'^\s*(?:<res>\s*)?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$'
        )
        pos = self._log_pos
        while time.time() - start < timeout:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(pos)
                    lines = f.readlines()
                    pos = f.tell()
                # 从后往前找最近的结果行，避免被其它输出打断
                for line in reversed(lines):
                    m = float2.match(line.strip())
                    if m:
                        srred = float(m.group(1))
                        trred = float(m.group(2))
                        self._log_pos = pos
                        return srred, trred
            time.sleep(0.05)
        self._log_pos = pos
        raise RuntimeError("等待 MATLAB 结果超时（日志未出现结果行）。")

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''

    def predict_video_source(self, vid_source, frame_padding="replicate"):
        h, w, N_frames = vid_source.get_video_size()

        Q = 0.0
        N = 0
        s_indexes, t_indexes = [], []

        Yr_prev = None
        Yt_prev = None

        max_lum = 255
        offset = 16
        weight = 219

        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            T_enc_np = T.squeeze().permute(1, 2, 0).cpu().numpy()
            R_enc_np = R.squeeze().permute(1, 2, 0).cpu().numpy()

            # RGB -> Y（Rec.709）
            YUV_t = (np.reshape(T_enc_np, (h * w, 3), order='F') @ self.col_mat.T).reshape(T_enc_np.shape, order='F')
            YUV_r = (np.reshape(R_enc_np, (h * w, 3), order='F') @ self.col_mat.T).reshape(R_enc_np.shape, order='F')

            Yt = (weight * YUV_t[:, :, 0] + offset).clip(0, max_lum)
            Yr = (weight * YUV_r[:, :, 0] + offset).clip(0, max_lum)

            # 统一在 matlab_cwd 里建临时 .mat，避免空格路径
            def _save_mat_and_send(frames, prefix):
                fd, abs_path = tempfile.mkstemp(prefix=prefix, suffix=".mat", dir=self.matlab_cwd)
                os.close(fd)
                basename = os.path.basename(abs_path)  # 只传文件名
                try:
                    scipy.io.savemat(abs_path, frames)
                    tmp_cmd = self.cmd_file + ".tmp"
                    with open(tmp_cmd, "w", encoding="utf-8") as f:
                        f.write(f"c {basename}\n")
                    os.replace(tmp_cmd, self.cmd_file)
                    return abs_path  # 供 finally 删除
                except Exception:
                    # 失败也尽量清掉
                    try:
                        os.remove(abs_path)
                    except OSError:
                        pass
                    raise

            if N_frames == 1:
                frames = {'Yr': Yr, 'Yt': Yt, 'Yr_prev': Yr, 'Yt_prev': Yt, 'type': 'image'}
                abs_path = _save_mat_and_send(frames, prefix="SPEEDQA-")
                try:
                    srred, trred = self._wait_for_result_in_log(timeout=10.0)
                    Q = srred
                    s_indexes.append(srred)
                    t_indexes.append(trred)
                    N = 1
                finally:
                    try:
                        os.remove(abs_path)
                    except OSError:
                        pass

            else:
                if ff > 0:
                    frames = {'Yr': Yr, 'Yt': Yt, 'Yr_prev': Yr_prev, 'Yt_prev': Yt_prev, 'type': 'video'}
                    abs_path = _save_mat_and_send(frames, prefix="STRRED-")
                    try:
                        srred, trred = self._wait_for_result_in_log(timeout=10.0)
                        Q_frame = srred * trred
                        s_indexes.append(srred)
                        t_indexes.append(trred)
                        Q += Q_frame
                        N += 1
                    finally:
                        try:
                            os.remove(abs_path)
                        except OSError:
                            pass

                Yr_prev = Yr
                Yt_prev = Yt

        if len(s_indexes) == 0 or len(t_indexes) == 0:
            return torch.as_tensor(float('nan')), None

        return torch.as_tensor(np.mean(s_indexes) * np.mean(t_indexes)), None

    def set_display_model(self, display_photometry, display_geometry):
        self.max_L = display_photometry.get_peak_luminance()
        self.max_L = np.array(min(self.max_L, 300))

    def short_name(self):
        return 'SpeedQA'

    def quality_unit(self):
        return None

    def get_info_string(self):
        return None

    @staticmethod
    def name():
        return "SpeedQA"

    @staticmethod
    def is_lower_better():
        return False

    @staticmethod
    def predictions_range():
        return 0, 100
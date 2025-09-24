# NonSmoothness_extroplate.py
# -*- coding: utf-8 -*-
"""
Extrapolated Non-smoothness test over Velocity (no disk I/O)
------------------------------------------------------------
- 固定 temporal frequencies 与 contrast 倍数；
- 在给定速度轴（0.02~200 deg/s）上外推；
- 直接“程序化生成”参考视频（sRGB，8-bit 量化前在浮点域做），不再从 mp4 读取；
- 用 CSF 计算 contrast_threshold，并据此生成 contrast_stimulus（若 >1 则丢弃）。

依赖：
  numpy, pandas, torch, pycvvdp
  你的 CSF 实现：CSF_castleCSF.py，类名 CSF_CastleCSF
  你的帧保持工具：tool.video_changing.frame_hold_like_reference
"""

import math
import numpy as np
import pandas as pd
import torch

from typing import Sequence, List

from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
from tool.video_changing import frame_hold_like_reference

# === 引入你的 CSF 实现 ===
from CSF_py.CSF_castleCSF import CSF_CastleCSF


# =========================
# 全局默认参数（可按需调整）
# =========================
# 屏幕物理尺寸（与你的视频生成脚本一致）
SCREEN_WIDTH_M  = 0.30
SCREEN_HEIGHT_M = 0.22
CYCLES_ACROSS_WIDTH = 4   # 30 cm 宽内 4 个周期

# 参考视频的像素分辨率/时长/帧率（在内存里生成）
WIDTH_PX  = round(480 / 4)   # 与你的示例一致：30 cm * 16 px/cm ≈ 300 px
HEIGHT_PX = round(352 / 4)   # ≈ 220 px
REFERENCE_FPS = 240           # 与原测试保持一致
DURATION_S = 2.0              # 每段参考视频时长
Y_PEAK  = 1000.0              # cd/m^2
L_MEAN  = 400.0               # cd/m^2

# 默认目标轴
DEFAULT_TEMPORAL_FREQS: Sequence[float] = [1.5, 3.0, 6.0, 12.0]
DEFAULT_TIMES: Sequence[int] = [3, 10, 30, 100]
DEFAULT_VELOCITIES = np.logspace(np.log10(0.02), np.log10(200.0), num=10)  # deg/s

# 常规模式下：采样频率列表（对数均匀 2~100 Hz）
DEFAULT_SAMPLING_FREQS = np.logspace(np.log10(2.0), np.log10(100.0), num=10)


# =========================
# sRGB 编码（线性→编码）
# =========================
def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """
    线性相对亮度（0~1） -> sRGB 编码（0~1）
    """
    a = 0.055
    x = np.clip(linear, 0.0, 1.0).astype(np.float32)
    low = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float32)
    out[low] = 12.92 * x[low]
    out[~low] = (1 + a) * np.power(x[~low], 1/2.4) - a
    return out


# =========================
# 在内存生成参考视频（sRGB）
# =========================
def generate_grating_video_tensor(
    f_t_hz: float,
    contrast: float,
    width_px: int = WIDTH_PX,
    height_px: int = HEIGHT_PX,
    fps: int = REFERENCE_FPS,
    duration_s: float = DURATION_S,
    screen_width_m: float = SCREEN_WIDTH_M,
    screen_height_m: float = SCREEN_HEIGHT_M,
    cycles_across_width: int = CYCLES_ACROSS_WIDTH,
    l_peak: float = Y_PEAK,
    l_mean: float = L_MEAN,
) -> torch.Tensor:
    """
    生成一个水平移动正弦光栅的 sRGB 张量，形状 [B,3,F,H,W]，范围 [0,1]。
    - 内容实现与你“视频导出脚本”一致，只是这里不落盘。
    - 频率在物理空间里固定为“屏幕宽度内 4 个周期”，因此与 viewing distance 无关。
    """
    # 每米像素（按宽度）
    px_per_m_w = width_px / screen_width_m

    # 屏幕上的空间频率（cycles/m），固定为 4 周期 / 屏幕宽
    f_s_c_per_m = cycles_across_width / screen_width_m  # e.g., 4 / 0.30 ≈ 13.333 c/m

    # 时间网格
    num_frames = int(round(duration_s * fps))
    t_array = np.arange(num_frames, dtype=np.float32) / float(fps)

    # 水平坐标（以“米”为单位；像素中心）
    x_px = (np.arange(width_px, dtype=np.float32) + 0.5)
    x_m  = x_px / px_per_m_w

    two_pi = 2.0 * math.pi
    phase_x = two_pi * f_s_c_per_m * x_m  # (W,)

    frames = np.empty((num_frames, height_px, width_px, 3), dtype=np.float32)

    for i, t in enumerate(t_array):
        phase_t = - two_pi * f_t_hz * t
        cos_line = np.cos(phase_x + phase_t, dtype=np.float32)  # (W,)
        # 线性亮度（nits）
        L_line = l_mean * (1.0 + contrast * cos_line)           # (W,)
        L_frame = np.tile(L_line[None, :], (height_px, 1))      # (H,W)
        # 限幅与归一化
        Lin01 = np.clip(L_frame / l_peak, 0.0, 1.0).astype(np.float32)
        # sRGB 编码
        sr = srgb_encode(Lin01)
        # 灰度到 RGB
        frames[i, :, :, 0] = sr
        frames[i, :, :, 1] = sr
        frames[i, :, :, 2] = sr

    # 变换为 [B,3,F,H,W]
    frames = np.transpose(frames, (3, 0, 1, 2))  # (C,F,H,W)
    return torch.tensor(frames[None, ...])       # (B,C,F,H,W)


# =========================
# 几何与单位换算
# =========================
def compute_distance_for_cpd(fs_cpd: float, screen_width_m: float = SCREEN_WIDTH_M,
                             cycles: int = CYCLES_ACROSS_WIDTH) -> float:
    """
    给定视角空间频率 Fs(cpd)，且屏幕宽度内固定 cycles 个周期。
    有：theta_x(deg) = cycles / Fs_cpd。由水平视角反推 viewing distance:
    distance = (screen_width/2) / tan(theta_x/2 in rad)
    """
    theta_x_deg = cycles / max(fs_cpd, 1e-6)
    phi = math.radians(theta_x_deg * 0.5)
    return (screen_width_m * 0.5) / max(math.tan(phi), 1e-6)


def compute_ppd(width_m: float, width_px: int, distance_m: float) -> float:
    """每像素多少度（ppd 的倒数），返回 ppd。"""
    # 单像素的水平视角（deg）
    pix_deg = 2.0 * math.degrees(math.atan(0.5 * width_m / width_px / distance_m))
    return 1.0 / max(pix_deg, 1e-9)


def diagonal_inches(width_m: float, height_m: float) -> float:
    return math.hypot(width_m, height_m) / 0.0254


# =========================
# 主测试类
# =========================
class NonSmoothnessExtrapolate:
    """
    与 NonSmoothnessSinGratingTest 接口兼容的 self-contained 测试类，
    但条件轴改为 velocity（外推），且参考视频在内存生成，不再读取 mp4。
    """

    def __init__(
        self,
        temporal_freqs: Sequence[float] = DEFAULT_TEMPORAL_FREQS,
        contrast_times: Sequence[int] = DEFAULT_TIMES,
        velocities_deg_s: np.ndarray = DEFAULT_VELOCITIES,
        sampling_freqs_hz: np.ndarray = DEFAULT_SAMPLING_FREQS,
        y_peak_cd_m2: float = Y_PEAK,
        avg_luminance_cd_m2: float = L_MEAN,
        device: str | torch.device | None = None,
        # 生成视频的画面与时序设置
        width_px: int = WIDTH_PX,
        height_px: int = HEIGHT_PX,
        ref_fps: int = REFERENCE_FPS,
        duration_s: float = DURATION_S,
    ):
        self.temporal_freqs = np.array(list(temporal_freqs), dtype=float)
        self.contrast_times = np.array(list(contrast_times), dtype=int)
        self.velocities = np.array(list(velocities_deg_s), dtype=float)
        self.sampling_freqs_hz = np.array(list(sampling_freqs_hz), dtype=float)

        self.y_peak = float(y_peak_cd_m2)
        self.avg_lum = float(avg_luminance_cd_m2)

        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.ref_fps = int(ref_fps)
        self.duration_s = float(duration_s)

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # photometry（sRGB EOTF，峰值 1000）
        self.display_photometry = vvdp_display_photo_eotf(
            Y_peak=self.y_peak, contrast=1000, source_colorspace='sRGB'
        )

        # 构造条件表 + CSF 过滤（contrast_stimulus <= 1）
        self._build_conditions()

        # 预览信息（与旧接口保持）
        self._preview_folder = 'nonsmoothness_sin_grating/extrapolated_velocity'
        self._preview_as = 'video'

    # --------- 条件构造（含 CSF）---------
    def _build_conditions(self):
        rows: List[dict] = []
        csf = CSF_CastleCSF()

        for ft in self.temporal_freqs:
            for times in self.contrast_times:
                for v in self.velocities:
                    fs_cpd = max(ft / max(v, 1e-9), 1e-6)  # Fs = Ft / V
                    dist_m = compute_distance_for_cpd(fs_cpd)
                    # 垂直视角与面积
                    theta_x = CYCLES_ACROSS_WIDTH / fs_cpd  # deg
                    theta_y = 2.0 * math.degrees(math.atan((SCREEN_HEIGHT_M * 0.5) / dist_m)) * 2.0 * 0.5

                    area_deg2 = max(theta_x * max(theta_y, 1e-9), 1e-9)

                    # CSF 敏感度与阈值
                    csf_pars = {
                        "s_frequency": float(fs_cpd),
                        "t_frequency": float(ft),
                        "orientation": 0.0,
                        "luminance": self.avg_lum,
                        "area": float(area_deg2),
                        "eccentricity": 0.0,
                    }
                    sens = float(csf.sensitivity(csf_pars)[0])
                    if not np.isfinite(sens) or sens <= 0:
                        continue
                    thr = 1.0 / sens
                    contrast = times * thr
                    if contrast > 1.0 or not np.isfinite(contrast):
                        # 超出可显示对比度，跳过
                        continue

                    for samp in self.sampling_freqs_hz:
                        rows.append({
                            "temporal_freq_Hz": float(ft),
                            "contrast_times": int(times),
                            "velocity_deg_s": float(v),
                            "spatial_freq_cpd": float(fs_cpd),
                            "distance_m": float(dist_m),
                            "area_deg2": float(area_deg2),
                            "contrast_threshold": float(thr),
                            "contrast_stimulus": float(contrast),
                            "sampling_freq_Hz": float(samp),
                        })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise RuntimeError("没有可用条件（可能全部 contrast_stimulus>1 被过滤）。请放宽速度范围或 times。")

        # 排序、去重（防止数值误差）
        df.sort_values(by=["temporal_freq_Hz", "contrast_times", "velocity_deg_s", "sampling_freq_Hz"],
                       inplace=True, ignore_index=True)
        self.conditions_df = df

        # 供绘图/size
        self.unique_tfs = np.sort(self.conditions_df['temporal_freq_Hz'].unique())
        self.unique_times = np.sort(self.conditions_df['contrast_times'].unique())
        self.unique_sampling = np.unique(self.conditions_df['sampling_freq_Hz'].values)

    # ------------------- 与旧接口兼容的方法 -------------------

    def __len__(self):
        return len(self.conditions_df)

    def short_name(self):
        return 'Non-smoothness - Sinusoidal Grating (Velocity Extrapolation, In-Memory)'

    def latex_name(self):
        return self.short_name()

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def get_row_header(self):
        return ['Sampling Frequency', 'Velocity', 'Temporal Frequency', 'Times']

    def get_rows_conditions(self):
        samp = torch.tensor(self.conditions_df['sampling_freq_Hz'].values, dtype=torch.float32)
        vel  = torch.tensor(self.conditions_df['velocity_deg_s'].values, dtype=torch.float32)
        tf   = torch.tensor(self.conditions_df['temporal_freq_Hz'].values, dtype=torch.float32)
        times= torch.tensor(self.conditions_df['contrast_times'].values, dtype=torch.int32)
        return [samp, vel, tf, times]

    def size(self):
        n_y = len(self.unique_sampling)
        n_total = len(self)
        n_x = max(1, n_total // n_y)
        return (n_y, n_x)

    def units(self):
        return ['log', 'linear'], ['Hz', 'deg/s']

    def get_ticks(self):
        y_ticks = torch.tensor([2, 3, 5, 10, 20, 50, 100], dtype=torch.float32)
        v = self.conditions_df['velocity_deg_s'].values
        v_min, v_max = float(np.min(v)), float(np.max(v))
        if v_min <= 0:
            v_min = 1e-3
        x_ticks = np.linspace(v_min, v_max, num=6, dtype=float)
        return torch.tensor(x_ticks, dtype=torch.float32), y_ticks

    # ------------------- 条件生成：返回(video_source, photometry, geometry) -------------------

    @staticmethod
    def _sampling_to_step(sampling_freq_hz: float, ref_fps: int) -> int:
        if sampling_freq_hz <= 0:
            return 1
        step = int(np.clip(int(round(ref_fps / float(sampling_freq_hz))), 1, max(1, ref_fps)))
        return step

    def _make_test_from_reference_srgb(self, ref_video_srgb: torch.Tensor, sampling_freq_hz: float) -> torch.Tensor:
        step = self._sampling_to_step(sampling_freq_hz, self.ref_fps)
        return frame_hold_like_reference(ref_video_srgb, step)

    def get_condition(self, index: int):
        """
        返回：
          (video_source, display_photometry, display_geometry)
        其中 video_source 的 (test, reference) 均为 sRGB，维度顺序 B,C,F,H,W。
        """
        with torch.no_grad():
            row = self.conditions_df.iloc[index]
            temporal_freq = float(row['temporal_freq_Hz'])
            contrast = float(row['contrast_stimulus'])
            distance_m = float(row['distance_m'])
            sampling_freq = float(row['sampling_freq_Hz'])

            # 1) 在内存生成“参考”视频（sRGB，[B,3,F,H,W]）
            ref_video_srgb = generate_grating_video_tensor(
                f_t_hz=temporal_freq,
                contrast=contrast,
                width_px=self.width_px,
                height_px=self.height_px,
                fps=self.ref_fps,
                duration_s=self.duration_s,
                screen_width_m=SCREEN_WIDTH_M,
                screen_height_m=SCREEN_HEIGHT_M,
                cycles_across_width=CYCLES_ACROSS_WIDTH,
                l_peak=self.y_peak,
                l_mean=self.avg_lum,
            ).to(self.device)

            _, _, _, H, W = ref_video_srgb.shape

            # 2) 显示几何
            width_m = SCREEN_WIDTH_M
            height_m = SCREEN_HEIGHT_M
            ppd = compute_ppd(width_m, W, distance_m)
            diag_in = diagonal_inches(width_m, height_m)
            cond_display_geometry = vvdp_display_geometry([int(W), int(H)], diagonal_size_inches=diag_in, ppd=ppd)

            # 3) 生成“测试”视频（frame-hold）
            test_video_srgb = self._make_test_from_reference_srgb(ref_video_srgb, sampling_freq)

            # 4) 打包为 video_source（注意顺序：test, reference）
            vs = video_source.video_source_array(
                test_video_srgb, ref_video_srgb, self.ref_fps, dim_order="BCFHW",
                display_photometry=self.display_photometry
            )

            return vs, self.display_photometry, cond_display_geometry


# 便于从命令行简单测试（不做评估）
if __name__ == "__main__":
    T = NonSmoothnessExtrapolate(
        temporal_freqs=[1.5, 3.0, 6.0, 12.0],
        contrast_times=[3, 10, 30, 100],
        velocities_deg_s=np.logspace(np.log10(0.02), np.log10(200), 10),
        sampling_freqs_hz=np.logspace(np.log10(2.0), np.log10(100.0), 10),
    )
    print(T.short_name(), "| Total conditions:", len(T))
    # 拿一个条件跑一下生成与几何
    _ = T.get_condition(0)
    print("One condition constructed successfully.")

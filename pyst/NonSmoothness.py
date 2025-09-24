# NonSmothness.py
# -*- coding: utf-8 -*-
"""
NonSmoothnessSinGratingTest
--------------------------------
一个“非平滑感”（采样伪影）测试，用现成的正弦光栅视频作为参考（sRGB），
并通过“sample-and-hold”方式生成不同 Sampling Frequency 的测试序列，
再交由各 metric 评分。

新增：
- 支持在初始化时外部指定 include_temporal_freqs / include_contrast_times 子集。
- 新增 is_alignment_score 模式：
  * 从 CSV 的某一列（base_sampling_col）读取“基准采样频率”，
    在其附近用 multiplier=logspace(0.5, 2, N_multiplier) 扩展得到 sampling_freq 轴。
  * get_row_header / get_rows_conditions 会多返回 'Alignment Multiplier' 一列。

CSV 必含列（你给出的实际列）：
  distance_m, spatial_freq_cpd(或 spatial_frequency_cpd), temporal_freq_Hz,
  contrast_times, contrast_stimulus, ...
如启用 is_alignment_score=True，还需包含基准采样频率列（默认名 'sampling_freq_Hz'，可通过 base_sampling_col 指定）

参考视频命名（与生成脚本一致）：
  {video_root}/grating_{tf:.1f}Hz_{contrast:.3f}contrast_{distance:.3f}distance_{fps}fps.mp4
"""
import os
import math
import numpy as np
import pandas as pd
import torch
import imageio.v3 as iio

from pycvvdp import video_source
from pycvvdp.display_model import vvdp_display_photo_eotf, vvdp_display_geometry
from tool.video_changing import frame_hold_like_reference

# ===================== 可按需修改的默认路径/参数 =====================

DEFAULT_VIDEO_ROOT = r'E:\Py_codes\temporal_distortions\Video_Generation\sinusoidal_grating_videos_gt_2_small'
DEFAULT_CSV_PATH = r'E:\Py_codes\temporal_distortions\matlab_codes_2025_8_12/stimulus_params_gt_merged.csv'
REFERENCE_FPS = 240

# 常规模式下：采样频率列表，对数分布于 [2, 100] Hz
DEFAULT_SAMPLING_FREQS = np.logspace(np.log10(2.0), np.log10(100.0), num=10)  # 可改为 25

# 默认目标（若未外部指定）
# DEFAULT_TARGET_TEMPORAL_FREQS = [3.0, 12.0]
# DEFAULT_TARGET_TIMES = [10, 100]
DEFAULT_TARGET_TEMPORAL_FREQS = [1.5, 3.0, 6.0, 12.0]
DEFAULT_TARGET_TIMES = [3, 10, 30, 100]

# 物理屏幕设置
SCREEN_WIDTH_M = 0.30  # 30 cm 宽


# ===================== 基础工具 =====================

def read_mp4_video_srgb(video_path: str) -> torch.Tensor:
    """
    读取 mp4 为 sRGB 张量，范围 [0,1]，形状 [B,3,F,H,W] (B=1)。
    不做任何线性化或色域映射。
    """
    frames = iio.imread(video_path)  # F,H,W,C (uint8)
    frames = frames.astype(np.float32) / 255.0
    frames_chw = np.transpose(frames, (3, 0, 1, 2))  # C,F,H,W
    return torch.tensor(frames_chw[None, ...])  # B,C,F,H,W


# ===================== 主测试类 =====================

class NonSmoothnessSinGratingTest:
    """
    一个 self-contained 的 Test 类，不继承 ContrastDetectionTest，
    但实现与【代码1】兼容的必要方法。
    """

    def __init__(
            self,
            csv_path: str = DEFAULT_CSV_PATH,
            video_root_path: str = DEFAULT_VIDEO_ROOT,
            sampling_freqs_hz: np.ndarray = DEFAULT_SAMPLING_FREQS,
            y_peak_cd_m2: float = 1000.0,
            avg_luminance_cd_m2: float = 400.0,  # 信息记录；显示模型使用 Y_peak
            device: str | torch.device | None = None,
            include_temporal_freqs: list[float] | None = None,
            include_contrast_times: list[int] | None = None,
            # ---- 新增：对齐模式参数 ----
            is_alignment_score: bool = False,
            N_multiplier: int = 10,
            base_sampling_col: str = 'sample_freq_Hz',
    ):
        """
        Parameters
        ----------
        include_temporal_freqs : list[float] | None
            若指定，则仅加载这些 temporal frequencies；否则用默认。
        include_contrast_times : list[int] | None
            若指定，则仅加载这些 contrast_times；否则用默认。
        is_alignment_score : bool
            若 True，则在 CSV 中每一行的“基准采样频率”附近用对数倍率展开 sampling_freq 轴。
        N_multiplier : int
            is_alignment_score=True 时倍率向量长度；倍率范围为 [0.5, 2.0]（对数均匀）。
        base_sampling_col : str
            is_alignment_score=True 时，CSV 中“基准采样频率”的列名（单位 Hz）。
        """
        self.csv_path = csv_path
        self.video_root_path = video_root_path
        self.is_alignment_score = bool(is_alignment_score)
        self.N_multiplier = int(N_multiplier)
        self.base_sampling_col = str(base_sampling_col)

        # 采样频率（常规模式使用；对齐模式会忽略这个数组）
        self.sampling_freqs_hz = np.array(sampling_freqs_hz, dtype=float)

        self.y_peak = float(y_peak_cd_m2)
        self.avg_lum = float(avg_luminance_cd_m2)

        # 子集筛选（未指定使用默认）
        self.include_temporal_freqs = (
            list(include_temporal_freqs) if include_temporal_freqs is not None
            else list(DEFAULT_TARGET_TEMPORAL_FREQS)
        )
        self.include_contrast_times = (
            list(include_contrast_times) if include_contrast_times is not None
            else list(DEFAULT_TARGET_TIMES)
        )

        # Photometry：sRGB EOTF，Y_peak=1000 cd/m^2
        self.display_photometry = vvdp_display_photo_eotf(
            Y_peak=self.y_peak, contrast=1000, source_colorspace='sRGB'
        )

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # --- 读取并筛选 CSV 条件 ---
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # 必要列检测
        required_cols = ['temporal_freq_Hz', 'contrast_stimulus', 'distance_m', 'contrast_times']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column '{col}'")

        # 空间频率列
        sf_col = None
        for cand in ['spatial_freq_cpd', 'spatial_frequency_cpd']:
            if cand in df.columns:
                sf_col = cand
                break
        if sf_col is None:
            raise ValueError(
                "CSV must contain spatial frequency column 'spatial_freq_cpd' (or 'spatial_frequency_cpd').")

        # 目标筛选（使用外部指定的子集）
        df = df[df['temporal_freq_Hz'].isin(self.include_temporal_freqs)]
        df = df[df['contrast_times'].isin(self.include_contrast_times)]

        # 为匹配视频文件名，数值四舍五入到合适精度
        df['temporal_freq_Hz_round'] = df['temporal_freq_Hz'].astype(float).round(1)
        df['contrast_stimulus_round'] = df['contrast_stimulus'].astype(float).round(3)
        df['distance_m_round'] = df['distance_m'].astype(float).round(3)

        # 速度（横轴）
        df['velocity_deg_s'] = df['temporal_freq_Hz'] / df[sf_col]

        # ---- 对齐模式：构造倍率向量 ----
        if self.is_alignment_score:
            # 倍率：logspace(0.5, 2.0)，长度 N_multiplier
            # 注意：用 math.log10() 来匹配你的接口
            self.multiplier = torch.logspace(
                math.log10(0.5), math.log10(2.0), steps=self.N_multiplier
            ).cpu().numpy().astype(float)

            # 需要基准采样频率列
            if self.base_sampling_col not in df.columns:
                raise ValueError(
                    f"is_alignment_score=True 需要 CSV 列 '{self.base_sampling_col}' 作为基准采样频率（Hz）。\n"
                    f"可通过参数 base_sampling_col 指定正确列名。"
                )

        # ---------- 展开条件 ----------
        expanded_rows = []
        if not self.is_alignment_score:
            # 常规模式：每行 × sampling_freqs_hz
            for idx, row in df.iterrows():
                for samp in self.sampling_freqs_hz:
                    expanded_rows.append({
                        'row_index': idx,
                        'temporal_freq_Hz': float(row['temporal_freq_Hz_round']),
                        'contrast_times': int(row['contrast_times']),
                        'contrast_stimulus': float(row['contrast_stimulus_round']),
                        'distance_m': float(row['distance_m_round']),
                        'spatial_freq_cpd': float(row[sf_col]),
                        'velocity_deg_s': float(row['velocity_deg_s']),
                        'sampling_freq_Hz': float(samp),
                    })
        else:
            # 对齐模式：每行 × multiplier（围绕 CSV 的基准采样频率）
            for idx, row in df.iterrows():
                base_samp = float(row[self.base_sampling_col])
                for m in self.multiplier:
                    samp = base_samp * float(m)
                    expanded_rows.append({
                        'row_index': idx,
                        'temporal_freq_Hz': float(row['temporal_freq_Hz_round']),
                        'contrast_times': int(row['contrast_times']),
                        'contrast_stimulus': float(row['contrast_stimulus_round']),
                        'distance_m': float(row['distance_m_round']),
                        'spatial_freq_cpd': float(row[sf_col]),
                        'velocity_deg_s': float(row['velocity_deg_s']),
                        'sampling_freq_Hz': float(samp),
                        'alignment_multiplier': float(m),
                    })

        self.conditions_df = pd.DataFrame(expanded_rows)
        self.conditions_df.sort_values(
            by=['temporal_freq_Hz', 'contrast_times', 'sampling_freq_Hz'],
            inplace=True, ignore_index=True
        )

        # 预览/元信息
        self._preview_folder = os.path.join('nonsmoothness_sin_grating', 'sampling_freq_vs_velocity')
        self._preview_as = 'video'

        # 基于过滤后的 conditions_df
        self.unique_tfs = np.sort(self.conditions_df['temporal_freq_Hz'].unique())
        self.unique_times = np.sort(self.conditions_df['contrast_times'].unique())
        self.unique_sampling = np.unique(self.conditions_df['sampling_freq_Hz'].values)

        self.ref_fps = REFERENCE_FPS

    # ------------------- 与【代码1】兼容的必要方法 -------------------

    def __len__(self):
        return len(self.conditions_df)

    def short_name(self):
        base = 'Non-smoothness - Sinusoidal Grating'
        if self.is_alignment_score:
            base += ' (Alignment)'
        return base

    def latex_name(self):
        return self.short_name()

    def get_preview_folder(self):
        return self._preview_folder

    def get_condition_file_format(self):
        return self._preview_as

    def get_row_header(self):
        """
        纵轴=Sampling Frequency（log），横轴=Velocity（linear）。
        对齐模式下，额外包含 'Alignment Multiplier'。
        """
        header = ['Sampling Frequency', 'Velocity', 'Temporal Frequency', 'Times']
        if self.is_alignment_score:
            header.append('Alignment Multiplier')
        return header

    def get_rows_conditions(self):
        samp = torch.tensor(self.conditions_df['sampling_freq_Hz'].values, dtype=torch.float32)
        vel = torch.tensor(self.conditions_df['velocity_deg_s'].values, dtype=torch.float32)
        tf = torch.tensor(self.conditions_df['temporal_freq_Hz'].values, dtype=torch.float32)
        times = torch.tensor(self.conditions_df['contrast_times'].values, dtype=torch.int32)
        outs = [samp, vel, tf, times]
        if self.is_alignment_score:
            align = torch.tensor(self.conditions_df['alignment_multiplier'].values, dtype=torch.float32)
            outs.append(align)
        return outs

    def size(self):
        # 提供二维形状占位（y=sampling，x=展开后的其余组合），
        # 实际绘图建议按 (tf, times) 过滤后 reshape
        n_y = len(self.unique_sampling)
        n_total = len(self)
        n_x = max(1, n_total // n_y)
        return (n_y, n_x)

    def units(self):
        # y=log（Hz），x=linear（deg/s）
        return ['log', 'linear'], ['Hz', 'deg/s']

    def get_ticks(self):
        y_ticks = torch.tensor([2, 3, 5, 10, 20, 50, 100], dtype=torch.float32)
        v = self.conditions_df['velocity_deg_s'].values
        v_min, v_max = np.min(v), np.max(v)
        if v_min <= 0:
            v_min = 1e-3
        x_ticks = np.linspace(v_min, v_max, num=6)
        return torch.tensor(x_ticks, dtype=torch.float32), y_ticks

    # ------------------- 条件生成：返回(video_source, photometry, geometry) -------------------

    def _reference_video_path(self, temporal_freq_hz: float, contrast: float, distance_m: float) -> str:
        fname = (
            f'grating_{temporal_freq_hz:.1f}Hz_'
            f'{contrast:.3f}contrast_'
            f'{distance_m:.3f}distance_'
            f'{self.ref_fps}fps.mp4'
        )
        return os.path.join(self.video_root_path, fname)

    @staticmethod
    def _sampling_to_step(sampling_freq_hz: float, ref_fps: int) -> int:
        if sampling_freq_hz <= 0:
            return 1
        step = int(np.clip(int(round(ref_fps / float(sampling_freq_hz))), 1, max(1, ref_fps)))
        return step

    @staticmethod
    def _compute_ppd_from_distance(width_m: float, width_px: int, distance_m: float) -> float:
        pix_deg = 2.0 * math.degrees(math.atan(0.5 * width_m / width_px / distance_m))  # 单像素视角（度）
        return 1.0 / pix_deg

    @staticmethod
    def _compute_diagonal_inches_from_wh(width_m: float, height_m: float) -> float:
        diag_m = math.hypot(width_m, height_m)
        return diag_m / 0.0254  # m -> inch

    def _make_test_from_reference_srgb(self, ref_video_srgb: torch.Tensor, sampling_freq_hz: float) -> torch.Tensor:
        step = self._sampling_to_step(sampling_freq_hz, self.ref_fps)
        return frame_hold_like_reference(ref_video_srgb, step)

    def get_condition(self, index: int):
        """
        返回：
          (video_source, display_photometry, display_geometry)
        """
        with torch.no_grad():
            row = self.conditions_df.iloc[index]

            temporal_freq = float(row['temporal_freq_Hz'])
            contrast = float(row['contrast_stimulus'])  # 仍按 CSV 指定的视频对比度加载
            distance_m = float(row['distance_m'])
            sampling_freq = float(row['sampling_freq_Hz'])

            # 1) 参考视频路径
            ref_path = self._reference_video_path(temporal_freq, contrast, distance_m)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference video not found: {ref_path}")

            # 2) 读取参考视频（sRGB，[B,3,F,H,W]）
            ref_video_srgb = read_mp4_video_srgb(ref_path).to(self.device)
            _, _, _, H, W = ref_video_srgb.shape

            # 3) 几何（由 distance 和分辨率计算）
            width_m = SCREEN_WIDTH_M
            height_m = width_m * (H / float(W))
            ppd = self._compute_ppd_from_distance(width_m, W, distance_m)
            diag_in = self._compute_diagonal_inches_from_wh(width_m, height_m)
            cond_display_geometry = vvdp_display_geometry([int(W), int(H)], diagonal_size_inches=diag_in, ppd=ppd)

            # 4) 生成测试视频（frame-hold，sRGB）
            test_video_srgb = self._make_test_from_reference_srgb(ref_video_srgb, sampling_freq)

            # 5) 打包为 pycvvdp 的 video_source（注意顺序：test, reference）
            vs = video_source.video_source_array(
                test_video_srgb, ref_video_srgb, self.ref_fps, dim_order="BCFHW",
                display_photometry=self.display_photometry
            )

            return vs, self.display_photometry, cond_display_geometry

    # ------------------- （可选）绘图占位 -------------------

    def plot(self, predictions, reverse_color_order=False, title=None, output_filename=None, axis=None,
             is_first_column=False, fontsize=18):
        """
        占位；基于 **筛选后的 conditions_df** 作图。
        """
        import matplotlib.pyplot as plt

        cond = self.get_rows_conditions()
        y = cond[0].cpu().numpy()  # sampling
        x = cond[1].cpu().numpy()  # velocity
        z = np.array(predictions).reshape(-1)

        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(6, 4))
        axis.set_xscale('linear')
        axis.set_yscale('log')
        axis.set_xlabel('Velocity [deg/s]', fontsize=fontsize)
        axis.set_ylabel('Sampling Frequency [Hz]', fontsize=fontsize)
        axis.set_title(title or self.short_name(), fontsize=fontsize + 1)
        axis.grid(alpha=0.4)

        sc = axis.scatter(x, y, c=z, s=20)
        plt.colorbar(sc, ax=axis, label='Metric Prediction')

        if output_filename is not None:
            plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        if axis is None:
            plt.close()

    def plotly(self, predictions, reverse_color_order=False, title='', fig_id=None):
        import numpy as np
        import json

        js_command = ""

        # 1. 预测值 reshape
        preds = np.array(predictions, dtype=float).reshape(-1)
        n = min(len(preds), len(self.conditions_df))
        cdf = self.conditions_df.iloc[:n].copy().reset_index(drop=True)
        cdf['pred'] = np.nan_to_num(preds[:n], nan=np.nan)  # NaN 保持为 NaN

        # 2. X/Y 条件
        tf_list = sorted(np.unique(cdf['temporal_freq_Hz']).tolist())
        times_list = sorted(np.unique(cdf['contrast_times']).tolist())

        R, C = len(tf_list), len(times_list)

        traces_js, axis_defs, annos = [], [], []
        colorscale = 'Viridis'
        reversescale = 'true' if reverse_color_order else 'false'

        def axis_suffix(i):
            return '' if i == 1 else str(i)

        # 3. Ground Truth CSV
        gt_csv = pd.read_csv(self.csv_path)
        if 'sample_freq_Hz' not in gt_csv.columns:
            raise ValueError("CSV 缺少列 'sample_freq_Hz'，无法绘制 GT 折线。")
        if 'velocity_deg_s' not in gt_csv.columns:
            sf_col = 'spatial_freq_cpd' if 'spatial_freq_cpd' in gt_csv.columns else (
                'spatial_frequency_cpd' if 'spatial_frequency_cpd' in gt_csv.columns else None
            )
            if sf_col is None:
                raise ValueError("CSV 缺少 'velocity_deg_s'，且无法从空间频率列计算。")
            gt_csv['velocity_deg_s'] = gt_csv['temporal_freq_Hz'].astype(float) / gt_csv[sf_col].astype(float)

        trace_idx = 0
        for r, tf in enumerate(tf_list, start=1):
            for c, tms in enumerate(times_list, start=1):
                sub = cdf[(cdf['temporal_freq_Hz'] == tf) & (cdf['contrast_times'] == tms)]
                if len(sub) == 0:
                    continue

                # ---- 预测点 ----
                x_s = sub['velocity_deg_s'].astype(float).values
                y_s = sub['sampling_freq_Hz'].astype(float).values
                z_s = sub['pred'].astype(float).values

                # 保证至少有变化
                if np.allclose(z_s.max(), z_s.min()):
                    z_s = z_s.copy()
                    z_s[0] += 1e-6

                # 按照 unique X, Y 重排成二维网格
                x_unique = np.sort(np.unique(x_s))
                y_unique = np.sort(np.unique(y_s))

                Z = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)
                for xi, xv in enumerate(x_unique):
                    for yi, yv in enumerate(y_unique):
                        mask = (x_s == xv) & (y_s == yv)
                        if np.any(mask):
                            Z[yi, xi] = np.nanmean(z_s[mask])

                # 转 Python list（NaN → null）
                Z_list = [[(None if not np.isfinite(v) else float(v)) for v in row] for row in Z]

                trace_idx += 1
                suf = axis_suffix(trace_idx)
                xa, ya = f'x{suf}', f'y{suf}'
                show_scale = (r == 1 and c == len(times_list))
                sub_title = f"TF = {tf:.1f} Hz | Times = {int(tms)}"

                # ---- 等高线 ----
                traces_js.append(f"""{{
      type: 'contour',
      x: {json.dumps([float(v) for v in x_unique])},
      y: {json.dumps([float(v) for v in y_unique])},
      z: {json.dumps(Z_list)},
      colorscale: '{colorscale}',
      reversescale: {reversescale},
      ncontours: 22,
      contours: {{ coloring: 'lines' }},
      line: {{ width: 2 }},
      xaxis: '{xa}',
      yaxis: '{ya}',
      showscale: {str(show_scale).lower()},
      colorbar: {{ title: 'Prediction' }},
      hovertemplate: 'Velocity=%{{x:.3g}} deg/s<br>Sampling=%{{y:.3g}} Hz<br>Pred=%{{z:.4g}}<extra></extra>'
    }}""")

                # ---- Ground Truth 折线 ----
                gt_mask = (np.abs(gt_csv['temporal_freq_Hz'].astype(float) - float(tf)) < 0.051) & \
                          (gt_csv['contrast_times'].astype(int) == int(tms))
                gt_sub = gt_csv.loc[gt_mask, ['velocity_deg_s', 'sample_freq_Hz']].astype(float).dropna()
                if len(gt_sub) > 0:
                    gt_sub = gt_sub.sort_values(by='velocity_deg_s')
                    x_gt = gt_sub['velocity_deg_s'].tolist()
                    y_gt = gt_sub['sample_freq_Hz'].tolist()

                    traces_js.append(f"""{{
      type: 'scatter',
      mode: 'lines',
      x: {json.dumps(x_gt)},
      y: {json.dumps(y_gt)},
      line: {{ color: 'red', width: 2 }},
      name: 'GT',
      xaxis: '{xa}',
      yaxis: '{ya}',
      hovertemplate: 'GT<br>Velocity=%{{x:.3g}} deg/s<br>Sampling=%{{y:.3g}} Hz<extra></extra>',
      showlegend: false
    }}""")

                # ---- 轴设置 ----
                x_title = "Velocity [deg/s]" if r == R else ""
                y_title = "Sampling Frequency [Hz]" if c == 1 else ""

                # axis_defs.append(
                #     f"'xaxis{suf}': {{ type:'log', range:[-2,3], "  # X 轴固定 0.01~1000
                #     f"title:{{text:'{x_title}', font:{{size:11}}}} }}"
                # )
                axis_defs.append(
                    f"'xaxis{suf}': {{ type:'log', range:[-1.523,2.477], "  # X 轴固定 0.01~1000
                    f"title:{{text:'{x_title}', font:{{size:11}}}} }}"
                )
                axis_defs.append(
                    f"'yaxis{suf}': {{ type:'log', range:[0.3010,2], "  # Y 轴固定 2~100 (log10(2)=0.3010, log10(100)=2)
                    f"title:{{text:'{y_title}', font:{{size:11}}}} }}"
                )

                annos.append({
                    "text": sub_title,
                    "xref": f"{xa} domain",
                    "yref": f"{ya} domain",
                    "x": 0.5, "y": 1.02, "yanchor": "bottom",
                    "showarrow": False, "font": {"size": 11}
                })

        # ---- Layout ----
        safe_title = (title or self.short_name()).replace('"', r'\"')
        layout_parts = ", ".join(axis_defs)
        js = f"""
    (function() {{
      var data = [
        {",\n    ".join(traces_js)}
      ];
      var layout = {{
        title: "{safe_title}",
        grid: {{ rows: {R}, columns: {C}, pattern: 'independent' }},
        margin: {{ t: 90, r: 20, b: 60, l: 70 }},
        width: 300 * {C},
        height: 300 * {R}, 
        {layout_parts},
        annotations: {json.dumps(annos)},
        font: {{ size: 12 }}
      }};
      var config = {{ responsive: true, displaylogo: false }};
      Plotly.newPlot("{fig_id}", data, layout, config);
    }})();
    """
        return js


# 便于从命令行简单测试（不做评估）
if __name__ == "__main__":
    # 示例1：常规模式
    T1 = NonSmoothnessSinGratingTest(
        include_temporal_freqs=[3.0, 12.0],
        include_contrast_times=[10, 100],
        is_alignment_score=False,
    )
    print(T1.short_name(), "| Total conditions:", len(T1))

    # 示例2：对齐模式（请确认 CSV 中存在 'sampling_freq_Hz' 列；否则改 base_sampling_col）
    T2 = NonSmoothnessSinGratingTest(
        include_temporal_freqs=[3.0, 12.0],
        include_contrast_times=[10, 100],
        is_alignment_score=True,
        N_multiplier=10,
        base_sampling_col='sampling_freq_Hz',  # ← 改成你 CSV 的基准采样频率列名
    )
    print(T2.short_name(), "| Total conditions:", len(T2))

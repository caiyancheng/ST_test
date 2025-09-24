import numpy as np
from typing import Dict, Any, IO

# 假设 CSFBase 就是你上一条消息里的实现
from CSF_py.CSF_base import CSFBase

class CSF_CastleCSF_Chrom(CSFBase):
    # ------- 常量 -------
    Y_min: float = 1e-3
    Y_max: float = 1e4
    rho_min: float = 2 ** -4
    rho_max: float = 64.0
    ecc_max: float = 120.0

    def __init__(self, colour: str):
        super().__init__()
        self.use_gpu: bool = True
        self.ps_beta: float = 1.0
        self.par: Dict[str, Any] = self.get_default_par(colour)

    # ------- 名称 -------
    def short_name(self) -> str:
        return "castle-csf-chrom"

    def full_name(self) -> str:
        return "castleCSF-chrom"

    # ------- 主接口：敏感度 -------
    def sensitivity(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        # 需要：luminance 或 lms_bkg；ge_sigma 或 area
        csf_pars = self.test_complete_params(csf_pars, requires=("luminance", "ge_sigma"))

        ecc = np.asarray(csf_pars["eccentricity"])
        sigma = np.asarray(csf_pars["ge_sigma"])
        rho = np.asarray(csf_pars["s_frequency"])
        omega = np.asarray(csf_pars.get("t_frequency", 0))
        lum = np.asarray(csf_pars["luminance"])

        R_sust = self.get_sust_trans_resp(omega)

        # 刺激面积（以高斯包络半径 sigma 表示）
        A = np.pi * (sigma ** 2)

        S_sust = self.csf_chrom(rho, A, lum, ecc, self.par["ch_sust"])

        S = R_sust * S_sust

        # 离心率引起的敏感度下降（window of visibility + 扩展）
        vis_field = np.asarray(csf_pars.get("vis_field", 180))
        alpha = np.minimum(1.0, np.abs(vis_field - 180.0) / 90.0)
        ecc_drop = alpha * self.par["ecc_drop"] + (1 - alpha) * self.par["ecc_drop_nasal"]
        ecc_drop_f = alpha * self.par["ecc_drop_f"] + (1 - alpha) * self.par["ecc_drop_f_nasal"]
        a = ecc_drop + rho * ecc_drop_f
        S = S * (10.0 ** (-a * ecc))

        return S

    # ------- “edge” 版本 -------
    def sensitivity_edge(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        csf_pars = dict(csf_pars)  # 浅拷贝
        csf_pars["s_frequency"] = np.logspace(np.log10(0.125), np.log10(16), 100)

        csf_pars = self.test_complete_params(csf_pars, requires=("luminance", "ge_sigma"))

        # 保存 ge_sigma（半径），随后以固定 area 计算
        radius = np.asarray(csf_pars["ge_sigma"]).reshape(-1)
        csf_pars.pop("ge_sigma", None)
        csf_pars["area"] = 3.09781  # 固定优化面积

        beta = 4.0
        S_gabor = self.sensitivity(csf_pars)  # 形状 ~ [freq, ...]（广播）
        S = S_gabor * (radius[np.newaxis, :] ** (1.0 / beta))
        S = np.max(S, axis=0)  # 对频率维取最大
        return S

    # ------- 持续通道时间响应 -------
    def get_sust_trans_resp(self, omega):
        omega = np.asarray(omega, dtype=float)
        sigma_sust = self.par["sigma_sust"]
        beta_sust = self.par["beta_sust"]
        R_sust = np.exp(-(omega ** beta_sust) / (sigma_sust))
        return R_sust

    # ------- 色度 CSF -------
    def csf_chrom(self, freq, area, lum, ecc, ch_pars: Dict[str, Any]) -> np.ndarray:
        freq = np.asarray(freq, dtype=float)
        area = np.asarray(area, dtype=float)
        lum = np.asarray(lum, dtype=float)
        ecc = np.asarray(ecc, dtype=float)  # 当前未直接使用

        S_max = self.get_lum_dep(ch_pars["S_max"], lum)
        f_max = self.get_lum_dep(ch_pars["f_max"], lum)
        bw = ch_pars["bw"]

        # --- 截断的色度 log-parabola ---
        # 广播：把 freq 放在第一维，f_max/L 在后续维度
        freq_shape = (freq.size,) + (1,) * np.asarray(f_max).ndim
        freq_b = freq.reshape(freq_shape)
        fmax_b = np.asarray(f_max)

        S_LP = 10.0 ** (-(np.abs(np.log10(freq_b) - np.log10(fmax_b)) ** 2.0) / (2.0 ** bw))

        # 计算每个 f_max 位置在 freq 维度上的最大值，并把 freq<f_max 的位置替换为这个最大值
        # 与 MATLAB: max_mat = repmat(max(S_LP), size(freq)); S_LP(freq<f_max)=max_mat(...)
        max_over_freq = S_LP.max(axis=0, keepdims=True)
        mask = (freq_b < fmax_b)
        S_LP = np.where(mask, max_over_freq, S_LP)

        # 顶点
        S_peak = S_max * S_LP  # 广播到同形状

        # --- Rovamo 尺寸模型 ---
        f0 = self.par.get("f_0", 0.65)
        A0 = self.par.get("A_0", 270.0)
        Ac = A0 / (1.0 + (freq_b / f0) ** 2.0)

        S = S_peak * np.sqrt(Ac / (1.0 + Ac / area)) * (freq_b)
        # 压回原广播后的形状（与 freq/f_max 广播后一致）
        return S

    # ------- 绘图描述（可选）-------
    def get_plot_description(self):
        pd = []
        pd.append({"title": "Sustained response", "id": "sust_trans"})
        pd.append({"title": "Peak sensitivity", "id": "peak_s"})
        return pd

    # ------- 绘图（可选）-------
    def plot_mechanism(self, plt_id: str):
        import matplotlib.pyplot as plt

        if plt_id == "sust_trans":
            omega = np.linspace(0, 100, 500)
            R_sust = self.get_sust_trans_resp(omega)
            plt.figure(figsize=(6, 6))
            plt.plot(omega, R_sust, label="Sustained")
            plt.xlabel("Temp. freq. [Hz]")
            plt.ylabel("Response")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

        elif plt_id == "peak_s":
            f = np.logspace(-2, np.log10(5), 1024)
            L = np.logspace(-2, 4, 200)
            LL, ff = np.meshgrid(L, f)

            plt.figure(figsize=(6, 6))
            csfpar = dict(
                luminance=LL.ravel(),
                s_frequency=ff.ravel(),
                t_frequency=0.0,
                area=np.pi * (1.5 ** 2),
                eccentricity=0.0,
                ge_sigma=1.5,  # 仅用于通过校验；实际 area 被使用
                vis_field=180.0,
            )
            S = self.sensitivity(csfpar).reshape(ff.shape)
            S_max = np.max(S, axis=0)
            plt.plot(L, S_max, label="0 Hz")

            L_dvr = np.logspace(-1, 0, 100)
            plt.plot(L_dvr, np.sqrt(L_dvr) * 50, "--", label="DeVries-Rose law")

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Luminance [cd/m²]")
            plt.ylabel("Peak sensitivity")
            plt.ylim(1, 1000)
            plt.grid(True, which="both", ls=":")
            plt.legend(loc="best")
            plt.show()
        else:
            raise ValueError("Wrong plt_id")

    # ------- 打印（覆写，仅打印 ch_sust 和顶层其他字段）-------
    def print(self, fh: IO[str]):
        for k, v in self.par["ch_sust"].items():
            fh.write(f"\t\t\t\tp.ch_sust.{k} = ")
            self.print_vector(fh, np.asarray(v))
            fh.write(";\n")
        fh.write("\n")
        for k, v in self.par.items():
            if k in ("ch_sust", "ds"):
                continue
            fh.write(f"\t\t\t\tp.{k} = ")
            self.print_vector(fh, np.asarray(v))
            fh.write(";\n")

    # ------- 默认参数（按颜色方向）-------
    @staticmethod
    def get_default_par(colour: str) -> Dict[str, Any]:
        p: Dict[str, Any] = CSFBase.get_dataset_par()

        if colour == "rg":
            p.update(
                dict(
                    ch_sust=dict(
                        S_max=[681.434, 38.0038, 0.480386],
                        f_max=0.0178364,
                        bw=2.42104,
                    ),
                    A_0=2816.44,
                    f_0=0.0711058,
                    sigma_sust=16.4325,
                    beta_sust=1.15591,
                    ecc_drop=0.0591402,
                    ecc_drop_nasal=2.89615e-05,
                    ecc_drop_f=2.04986e-69,
                    ecc_drop_f_nasal=0.18108,
                )
            )
        elif colour == "yv":
            p.update(
                dict(
                    ch_sust=dict(
                        S_max=[166.683, 62.8974, 0.41193],
                        f_max=0.00425753,
                        bw=2.68197,
                    ),
                    A_0=2.82789e07,
                    f_0=0.000635093,
                    sigma_sust=7.15012,
                    beta_sust=0.969123,
                    ecc_drop=0.00356865,
                    ecc_drop_nasal=5.85804e-141,
                    ecc_drop_f=0.00806631,
                    ecc_drop_f_nasal=0.0110662,
                )
            )
        else:
            raise ValueError("Invalid colour direction supplied (use 'rg' or 'yv')")

        return p

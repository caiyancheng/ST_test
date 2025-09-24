import numpy as np
from typing import Dict, Any
# 假设 CSFBase 已按上一条消息提供
from CSF_py.CSF_base import CSFBase

class CSF_StelaCSF_Lum_Peak(CSFBase):
    # -------- 常量 --------
    Y_min: float = 1e-3
    Y_max: float = 1e4
    rho_min: float = 2 ** -4
    rho_max: float = 64.0
    ecc_max: float = 120.0

    def __init__(self):
        super().__init__()
        self.use_gpu: bool = True
        self.ps_beta: float = 1.0
        self.par: Dict[str, Any] = self.get_default_par()

    # -------- 名称 --------
    def short_name(self) -> str:
        return "stela-csf-lum-peak"

    def full_name(self) -> str:
        return "stelaCSF_lum_peak"

    # -------- 主接口：敏感度 --------
    def sensitivity(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        # 需要：luminance 或 lms_bkg；ge_sigma 或 area
        csf_pars = self.test_complete_params(csf_pars, requires=("luminance", "ge_sigma"))

        ecc = np.asarray(csf_pars["eccentricity"])
        sigma = np.asarray(csf_pars["ge_sigma"])
        rho = np.asarray(csf_pars["s_frequency"])
        omega = np.asarray(csf_pars.get("t_frequency", 0))
        lum = np.asarray(csf_pars["luminance"])

        R_sust, R_trans = self.get_sust_trans_resp(omega, lum)

        # 刺激面积（高斯包络半径）
        A = np.pi * (sigma ** 2)

        S_sust = self.csf_achrom(rho, A, lum, ecc, self.par["ach_sust"])
        S_trans = self.csf_achrom(rho, A, lum, ecc, self.par["ach_trans"])

        S_aux = 0.0
        pm_ratio = 1.0

        if self.ps_beta != 1.0:
            beta = self.ps_beta
            S = (
                (R_sust * S_sust * np.sqrt(pm_ratio)) ** beta
                + (R_trans * S_trans * np.sqrt(1.0 / pm_ratio)) ** beta
                + (S_aux) ** beta
            ) ** (1.0 / beta)
        else:
            S = (
                R_sust * S_sust * np.sqrt(pm_ratio)
                + R_trans * S_trans * np.sqrt(1.0 / pm_ratio)
                + S_aux
            )

        # 离心率引起的敏感度下降（window of visibility + 扩展）
        vis_field = np.asarray(csf_pars.get("vis_field", 180))
        alpha = np.minimum(1.0, np.abs(vis_field - 180.0) / 90.0)
        ecc_drop = alpha * self.par["ecc_drop"] + (1 - alpha) * self.par["ecc_drop_nasal"]
        ecc_drop_f = alpha * self.par["ecc_drop_f"] + (1 - alpha) * self.par["ecc_drop_f_nasal"]
        a = ecc_drop + rho * ecc_drop_f
        S = S * (10.0 ** (-a * ecc))

        return S

    # -------- “edge” 版本 --------
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
        # 与 MATLAB: S_gabor .* (radius'.^(1/beta)) 对齐
        S = S_gabor * (radius[np.newaxis, :] ** (1.0 / beta))
        # 按频率维（第一维）取最大
        S = np.max(S, axis=0)

        # MATLAB 里还有 permute/circshift，这里通常已不需要
        return S

    # -------- 时间响应：持续/瞬时 --------
    def get_sust_trans_resp(self, omega, lum):
        omega = np.asarray(omega, dtype=float)
        lum = np.asarray(lum, dtype=float)

        sigma_sust = self.par["sigma_sust"]
        beta_sust = 1.3314

        omega_0 = np.log10(lum) * self.par["omega_trans_sl"] + self.par["omega_trans_c"]

        beta_trans = 0.1898
        sigma_trans = self.par["sigma_trans"]

        R_sust = np.exp(-(omega ** beta_sust) / (sigma_sust))
        R_trans = np.exp(-np.abs(omega ** beta_trans - omega_0 ** beta_trans) ** 2 / (sigma_trans))
        return R_sust, R_trans

    # -------- 无色通道 CSF --------
    def csf_achrom(self, freq, area, lum, ecc, ach_pars: Dict[str, Any]) -> np.ndarray:
        freq = np.asarray(freq, dtype=float)
        area = np.asarray(area, dtype=float)
        lum = np.asarray(lum, dtype=float)
        ecc = np.asarray(ecc, dtype=float)  # 目前未直接使用，但保持签名一致

        S_max = self.get_lum_dep(ach_pars["S_max"], lum)
        f_max = self.get_lum_dep(ach_pars["f_max"], lum)
        bw = ach_pars["bw"]
        a = ach_pars["a"]

        # 截断 log-parabola
        S_LP = 10.0 ** (-(np.log10(freq) - np.log10(f_max)) ** 2.0 / (2.0 ** bw))
        ss = (freq < f_max) & (S_LP < (1 - a))
        S_LP = np.where(ss, 1 - a, S_LP)

        S_peak = S_max * S_LP

        # Rovamo 尺寸模型
        f0 = ach_pars.get("f_0", 0.65)
        A0 = ach_pars.get("A_0", 270.0)
        Ac = A0 / (1.0 + (freq / f0) ** 2.0)

        S = S_peak * np.sqrt(Ac / (1.0 + Ac / area)) * (freq ** 1.0)
        return S

    # --------（可选）绘图描述 --------
    def get_plot_description(self):
        pd = []
        pd.append({"title": "Sustained and transient response", "id": "sust_trans"})
        pd.append({"title": "Peak sensitivity", "id": "peak_s"})
        return pd

    # --------（可选）绘图 --------
    def plot_mechanism(self, plt_id: str):
        import matplotlib.pyplot as plt

        if plt_id == "sust_trans":
            omega = np.linspace(0, 100, 500)
            lums = [0.1, 30, 1000]

            plt.figure(figsize=(6, 6))
            R_sust, _ = self.get_sust_trans_resp(omega, lums[0])
            plt.plot(omega, R_sust, label="Sustained")
            for L in lums:
                _, R_trans = self.get_sust_trans_resp(omega, L)
                plt.plot(omega, R_trans, label=f"Transient ({L} cd/m²)")
            plt.xlabel("Temp. freq. [Hz]")
            plt.ylabel("Response")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

        elif plt_id == "peak_s":
            f = np.logspace(-2, np.log10(5), 1024)
            L = np.logspace(-2, 4, 200)
            LL, ff = np.meshgrid(L, f)
            OMEGAs = [0, 5, 16]

            plt.figure(figsize=(6, 6))
            for omg in OMEGAs:
                csfpar = dict(
                    luminance=LL.ravel(),
                    s_frequency=ff.ravel(),
                    t_frequency=omg,
                    area=np.pi * (1.5 ** 2),
                    eccentricity=0,
                    ge_sigma=1.5,  # 仅用于通过校验；实际 area 被使用
                    vis_field=180,
                )
                S = self.sensitivity(csfpar).reshape(ff.shape)
                S_max = np.max(S, axis=0)
                plt.plot(L, S_max, label=f"{omg} Hz")

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

    # -------- 默认参数 --------
    @staticmethod
    def get_default_par() -> Dict[str, Any]:
        p: Dict[str, Any] = CSFBase.get_dataset_par()

        p.update(
            {
                "ach_sust": {
                    "S_max": [56.4947, 7.54726, 0.144532, 5.58341e-07, 9.66862e09],
                    "f_max": [1.78119, 91.5718, 0.256682],
                    "bw": 0.000213047,
                    "a": 0.100207,
                    "A_0": 157.103,
                    "f_0": 0.702338,
                },
                "ach_trans": {
                    "S_max": [0.193434, 2748.09],
                    "f_max": 0.000316696,
                    "bw": 2.6761,
                    "a": 0.000241177,
                    "A_0": 3.81611,
                    "f_0": 3.01389,
                },
                "sigma_trans": 0.0844836,
                "sigma_sust": 10.5795,
                "omega_trans_sl": 2.41482,
                "omega_trans_c": 4.7036,
                "ecc_drop": 0.0239853,
                "ecc_drop_nasal": 0.0400662,
                "ecc_drop_f": 0.0189038,
                "ecc_drop_f_nasal": 0.00813619,
            }
        )
        return p

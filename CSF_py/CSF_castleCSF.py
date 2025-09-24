import numpy as np
from typing import Dict, Any, Tuple, IO

from CSF_py.CSF_base import CSFBase
from CSF_py.CSF_stelaCSF_lum_peak import CSF_StelaCSF_Lum_Peak
from CSF_py.CSF_castleCSF_chrom import CSF_CastleCSF_Chrom

class CSF_CastleCSF(CSFBase):
    """
    Colour, Area, Spatial frequency, Temporal frequency, Luminance,
    Eccentricity dependent CSF (castleCSF)
    """

    # ---- 常量（类属性）----
    Mones = np.array([[1, 1, 0],
                      [1, 0, 0],
                      [1, 1, 0]], dtype=int)

    # Wuerger 2020 (JoV) 用的 LMS->DKL 机制矩阵参数（填到 Mones 为 0 的位置，列优先）
    colmat = np.array([2.3112, 0.0, 0.0, 50.9875], dtype=float)
    chrom_ch_beta: float = 2.0

    Y_min: float = 1e-3
    Y_max: float = 1e4
    rho_min: float = 2 ** -4
    rho_max: float = 64.0
    ecc_max: float = 120.0

    def __init__(self):
        super().__init__()
        self.use_gpu: bool = True
        self.ps_beta: float = 1.0

        # 组件模型
        self.castleCSF_ach = CSF_StelaCSF_Lum_Peak()
        self.castleCSF_rg  = CSF_CastleCSF_Chrom('rg')
        self.castleCSF_yv  = CSF_CastleCSF_Chrom('yv')

        self.par: Dict[str, Any] = self.get_default_par()
        self.update_parameters()

    # ---- 名称 ----
    def short_name(self) -> str:
        return "castle-csf"

    def full_name(self) -> str:
        return "castleCSF"

    # ---- 颜色机制矩阵 ----
    def get_lms2acc(self) -> np.ndarray:
        """
        返回 3x3 机制矩阵（ach, rg, yv）。按列优先将 colmat/自定义 par.colmat
        填入 Mones==0 的位置，然后乘以符号矩阵。
        """
        M = np.ones((3, 3), dtype=float)
        vals = np.array(self.par.get("colmat", self.colmat), dtype=float).ravel()
        maskF = (self.Mones == 0).flatten(order="F")
        M_flatF = M.flatten(order="F")
        if vals.size != maskF.sum():
            raise ValueError("colmat 的长度必须等于 Mones 中 0 的个数")
        M_flatF[maskF] = vals
        M = M_flatF.reshape((3, 3), order="F")

        signs = np.array([[ 1,  1, 1],
                          [ 1, -1, 1],
                          [-1, -1, 1]], dtype=float)
        return M * signs

    # ---- 主接口：敏感度 ----
    def sensitivity(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        # 需要 lms_bkg / ge_sigma
        csf_pars = self.test_complete_params(csf_pars, requires=("lms_bkg", "ge_sigma"))

        k = self.det_threshold(csf_pars)  # 标量或张量（去掉颜色维）
        lms_delta = np.asarray(csf_pars["lms_delta"], dtype=float)
        lms_bkg   = np.asarray(csf_pars["lms_bkg"],   dtype=float)

        # 门限处增量：k * lms_delta  (广播到最后一维=3)
        LMS_delta_thr = k[..., np.newaxis] * lms_delta

        # 以 RMS(LMS 对比) 定义的灵敏度（与 MATLAB 一致）
        contrast = LMS_delta_thr / lms_bkg
        rms = np.sqrt(np.sum(contrast**2, axis=-1)) / np.sqrt(3.0)
        S = 1.0 / rms
        return S

    # ---- 返回阈值系数 k（使得 k*lms_delta 为阈值）----
    def det_threshold(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        csf_pars = self.test_complete_params(csf_pars, requires=("lms_bkg", "ge_sigma"))

        lms_bkg   = np.asarray(csf_pars["lms_bkg"],   dtype=float)
        lms_delta = np.asarray(csf_pars["lms_delta"], dtype=float)

        C_A, C_R, C_Y = self.csf_chrom_directions(lms_bkg, lms_delta)

        # 组件 CSF（与 MATLAB 一致，直接用同一组 csf_pars 调子模型）
        C_A_n = C_A * self.castleCSF_ach.sensitivity(csf_pars)
        C_R_n = C_R * self.castleCSF_rg.sensitivity(csf_pars)
        C_Y_n = C_Y * self.castleCSF_yv.sensitivity(csf_pars)

        beta = self.chrom_ch_beta
        C = (C_A_n**beta + C_R_n**beta + C_Y_n**beta) ** (1.0 / beta)

        k_thr = C ** (-1.0)
        return k_thr

    # ---- 圆盘（edge）灵敏度 ----
    def sensitivity_edge(self, csf_pars: Dict[str, Any]) -> np.ndarray:
        """
        基于 “Modeling contrast sensitivity of discs” 的圆盘灵敏度。
        约束：不能同时指定 s_frequency 或 area。
        """
        if ("s_frequency" in csf_pars) or ("area" in csf_pars):
            raise ValueError("计算圆盘灵敏度时不能指定 s_frequency 或 area")

        csf_pars = self.test_complete_params(csf_pars, requires=("luminance", "ge_sigma"))

        # 给所有非标量参数增加一个前导维度（频率维）
        pars = dict(csf_pars)
        for k, v in list(pars.items()):
            v = np.asarray(v)
            if v.size > 1:
                pars[k] = v.reshape((1,) + v.shape)

        # 频率采样（前导维）
        pars["s_frequency"] = np.logspace(np.log10(0.125), np.log10(16), 64).reshape((-1, 1))

        # 保存半径并改用固定 area
        radius = np.asarray(pars["ge_sigma"], dtype=float)
        pars.pop("ge_sigma", None)

        # 论文中的优化参数（可被 self.par 覆盖）
        area_fixed = self.par.get("disc_area", 2.42437)
        beta = self.par.get("disc_beta", 3.01142)
        pars["area"] = area_fixed

        S_gabor = self.sensitivity(pars)
        S1 = S_gabor * (radius ** (1.0 / beta))
        # 沿频率维（axis=0）取最大，去掉频率维
        S = np.max(S1, axis=0)
        return S

    # ---- 颜色方向权重（ach/rg/yv）----
    def csf_chrom_directions(self, LMS_mean: np.ndarray, LMS_delta: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于机制矩阵把 LMS (mean, delta) 投影到 ACC(ach, rg, yv)，
        返回各方向上的归一化权重 C_A, C_R, C_Y（去掉颜色维的形状）。
        """
        LMS_mean = np.asarray(LMS_mean, dtype=float)
        LMS_delta = np.asarray(LMS_delta, dtype=float)

        # 计算广播后的总体形状，最后一维必须是 3 (L,M,S)
        bshape = np.broadcast_shapes(LMS_mean.shape, LMS_delta.shape)
        if bshape[-1] != 3:
            raise AssertionError("LMS 向量的最后一维必须为 3")
        target_size = bshape[:-1]

        M = self.get_lms2acc()  # 3x3

        # 广播 -> (-1,3) 做矩阵乘，再恢复形状
        L_b = np.broadcast_to(LMS_mean, bshape).reshape((-1, 3))
        D_b = np.broadcast_to(LMS_delta, bshape).reshape((-1, 3))

        ACC_mean = np.abs(L_b @ M.T).reshape(bshape)
        ACC_delta = np.abs(D_b @ M.T).reshape(bshape)

        # 取通道
        A_mean = ACC_mean[..., 0]
        R_mean = ACC_mean[..., 1]
        Y_mean = ACC_mean[..., 2]
        A_dlt  = ACC_delta[..., 0]
        R_dlt  = ACC_delta[..., 1]
        Y_dlt  = ACC_delta[..., 2]

        alpha = 0.0
        C_A = np.abs(A_dlt / (A_mean + 1e-12))
        C_R = np.abs(R_dlt / (alpha * R_mean + (1 - alpha) * A_mean + 1e-12))
        C_Y = np.abs(Y_dlt / (alpha * Y_mean + (1 - alpha) * A_mean + 1e-12))

        # 去掉颜色维后的形状（与 MATLAB 的 reshape_fix 一致）
        C_A = C_A.reshape(target_size)
        C_R = C_R.reshape(target_size)
        C_Y = C_Y.reshape(target_size)
        return C_A, C_R, C_Y

    # ---- 绘图描述（可选）----
    def get_plot_description(self):
        pd = []
        pd.append({"title": "Color mechanisms", "id": "col_mech"})
        pd.append({"title": "Sustained and transient response", "id": "sust_trans"})
        pd.append({"title": "Peak sensitivity", "id": "peak_s"})
        return pd

    # ---- 机制可视化（可选）----
    def plot_mechanism(self, plt_id: str):
        import matplotlib.pyplot as plt
        if plt_id == "col_mech":
            # 需要 lms2dkl_d65 函数；这里给出基础向量可视化（跳过 DKL）
            M = self.get_lms2acc()
            cm_lms = np.linalg.inv(M) @ np.eye(3)
            mech_label = ['achromatic', 'red-green', 'violet-yellow']
            plt.figure(figsize=(10, 4))
            for pp in range(2):
                ax = plt.subplot(1, 2, pp + 1)
                if pp == 0:
                    dd = (1, 2)  # 简单画 (x,y) 两维
                else:
                    dd = (1, 0)
                for cc in range(3):
                    ax.quiver(0, 0, cm_lms[dd[0], cc], cm_lms[dd[1], cc], angles='xy', scale_units='xy', scale=1,
                              label=mech_label[cc])
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_aspect('equal')
                ax.legend(loc='lower left')
            plt.show()

        elif plt_id == "sust_trans":
            omega = np.linspace(0, 60, 600)
            lums = [0.1, 30, 1000]

            plt.figure(figsize=(7, 6))
            hh = []
            for i, L in enumerate(lums):
                R_sust, R_trans = self.castleCSF_ach.get_sust_trans_resp(omega, L)
                if i == 0:
                    h0, = plt.plot(omega, R_sust, '-k', label='Sustained (achromatic)')
                    hh.append(h0)
                h, = plt.plot(omega, R_trans, '--', label=f'Transient (achromatic) ({L} cd/m²)')
                hh.append(h)

            R_sust_rg = self.castleCSF_rg.get_sust_trans_resp(omega)
            hh.append(plt.plot(omega, R_sust_rg, '-r', label='Sustained (red-green)')[0])

            R_sust_yv = self.castleCSF_yv.get_sust_trans_resp(omega)
            hh.append(plt.plot(omega, R_sust_yv, color=(0.6, 0, 1), label='Sustained (yellow-violet)')[0])

            plt.xlabel('Temp. freq. [Hz]')
            plt.ylabel('Response')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()

        elif plt_id == "peak_s":
            self.castleCSF_ach.plot_mechanism("peak_s")
        else:
            raise ValueError("Wrong plt_id")

    # ---- 参数设置/同步 ----
    def set_pars(self, pars_vector: np.ndarray):
        super().set_pars(pars_vector)
        self.update_parameters()
        return self

    def update_parameters(self):
        """把 self.par 中的组参数同步到各组件模型。"""
        if "ach" in self.par:
            self.castleCSF_ach.par = CSFBase.update_struct(self.par["ach"], self.castleCSF_ach.par)
        if "rg" in self.par:
            self.castleCSF_rg.par = CSFBase.update_struct(self.par["rg"], self.castleCSF_rg.par)
        if "yv" in self.par:
            self.castleCSF_yv.par = CSFBase.update_struct(self.par["yv"], self.castleCSF_yv.par)

    # ---- 打印参数（包含组件）----
    def print(self, fh: IO[str] = None):
        import sys
        if fh is None:
            fh = sys.stdout

        M = self.get_lms2acc()
        fh.write("M_lms2acc =\n")
        fh.write(str(M) + "\n\n")

        # 打印顶层参数
        super().print(fh)

        fh.write("Parameters for Ach component:\n")
        self.castleCSF_ach.print(fh)
        fh.write("\nParameters for RG component:\n")
        self.castleCSF_rg.print(fh)
        fh.write("\nParameters for YV component:\n")
        self.castleCSF_yv.print(fh)
        fh.write("\n")

    # ---- 默认参数 ----
    @staticmethod
    def get_default_par() -> Dict[str, Any]:
        p: Dict[str, Any] = CSFBase.get_dataset_par()

        p["rg"] = dict(
            sigma_sust=16.4325,
            beta_sust=1.15591,
            ch_sust=dict(
                S_max=[681.434, 38.0038, 0.480386],
                f_max=0.0178364,
                bw=2.42104,
            ),
            A_0=2816.44,
            f_0=0.0711058,
            ecc_drop=0.0591402,
            ecc_drop_nasal=2.89615e-05,
            ecc_drop_f=2.04986e-69,
            ecc_drop_f_nasal=0.18108,
        )

        p["yv"] = dict(
            sigma_sust=7.15012,
            beta_sust=0.969123,
            ch_sust=dict(
                S_max=[166.683, 62.8974, 0.41193],
                f_max=0.00425753,
                bw=2.68197,
            ),
            A_0=2.82789e07,
            f_0=0.000635093,
            ecc_drop=0.00356865,
            ecc_drop_nasal=5.85804e-141,
            ecc_drop_f=0.00806631,
            ecc_drop_f_nasal=0.0110662,
        )

        p["ach"] = dict(
            ach_sust=dict(
                S_max=[56.4947, 7.54726, 0.144532, 5.58341e-07, 9.66862e09],
                f_max=[1.78119, 91.5718, 0.256682],
                bw=0.000213047,
                a=0.100207,
                A_0=157.103,
                f_0=0.702338,
            ),
            ach_trans=dict(
                S_max=[0.193434, 2748.09],
                f_max=0.000316696,
                bw=2.6761,
                a=0.000241177,
                A_0=3.81611,
                f_0=3.01389,
            ),
            sigma_trans=0.0844836,
            sigma_sust=10.5795,
            omega_trans_sl=2.41482,
            omega_trans_c=4.7036,
            ecc_drop=0.0239853,
            ecc_drop_nasal=0.0400662,
            ecc_drop_f=0.0189038,
            ecc_drop_f_nasal=0.00813619,
        )

        return p

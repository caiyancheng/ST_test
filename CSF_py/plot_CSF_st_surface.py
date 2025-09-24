import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps as cmaps  # 取代 cm.get_cmap
from mpl_toolkits.mplot3d import Axes3D  # 仅用于激活 3D

# 如果这几个类不在同一文件，请按你的工程结构正确导入
from CSF_base import CSFBase
from CSF_stelaCSF_lum_peak import CSF_StelaCSF_Lum_Peak
from CSF_castleCSF_chrom import CSF_CastleCSF_Chrom
from CSF_castleCSF import CSF_CastleCSF


def CSF_spatial_temporal_plot(
    csf_model: CSFBase | None = None,
    s_range: tuple[float, float] = (0.1, 64.0),
    t_range: tuple[float, float] = (0.1, 32.0),
    n_s: int = 100,
    n_t: int = 80,
    luminance: float = 100.0,
    area: float = 100.0,
    eccentricity: float = 0.0,
    orientation: float = 0.0,
    levels: int = 15,  # 目前未使用，仅保留接口
    figsize: tuple[float, float] = (12.5, 5.5),
    cmap: str = "viridis",
):
    """
    左：对数轴热力图（线性 sensitivity 着色）
    右：3D 曲面（Z 为 log 视觉，但刻度与色条显示线性 sensitivity）

    返回:
        s_freqs, t_freqs, S
    """
    if csf_model is None:
        csf_model = CSF_CastleCSF()

    # 1) 对数采样
    s_min, s_max = s_range
    t_min, t_max = t_range
    s_freqs = np.logspace(np.log10(s_min), np.log10(s_max), n_s)
    t_freqs = np.logspace(np.log10(t_min), np.log10(t_max), n_t)

    # 2) 计算敏感度矩阵
    S = np.empty((n_t, n_s), dtype=float)
    for i, tf in enumerate(t_freqs):
        csf_pars = {
            "s_frequency": s_freqs,
            "t_frequency": float(tf),
            "orientation": orientation,
            "luminance": luminance,
            "area": area,
            "eccentricity": eccentricity,
        }
        S[i, :] = csf_model.sensitivity(csf_pars)

    # —— 线性着色，Z 轴 log 视觉需要正值 ——
    finite_mask = np.isfinite(S)
    if not np.any(finite_mask):
        raise ValueError("All sensitivities are NaN/Inf; nothing to plot.")

    S_max = float(np.nanmax(S[finite_mask]))
    S_lin = np.array(S, dtype=float)
    S_lin[~finite_mask] = 0.0
    S_lin[S_lin < 0] = 0.0

    tiny = max(S_max, 1.0) * 1e-12 if S_max > 0 else 1e-12
    S_pos = np.clip(S_lin, tiny, S_max if S_max > tiny else tiny * 10)
    Z_log = np.log10(S_pos)

    Sx, Ty = np.meshgrid(s_freqs, t_freqs)

    # 3) 布局
    fig = plt.figure(figsize=figsize, constrained_layout=True)  # 收紧左右子图间距
    fig.set_constrained_layout_pads(wspace=0.01, hspace=0.01, w_pad=0.01, h_pad=0.01)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    # 刻度
    base_ticks = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64]
    s_ticks = [v for v in base_ticks if s_min <= v <= s_max]
    t_ticks = [v for v in base_ticks if t_min <= v <= t_max]

    cmap_obj = cmaps.get_cmap(cmap)
    norm_lin = Normalize(vmin=0.0, vmax=S_max if S_max > 0 else 1.0)

    # ——— 左：热力图（只保留“外置”颜色条；删除 inset_axes） ———
    ax0 = fig.add_subplot(gs[0, 0])
    pcm = ax0.pcolormesh(Sx, Ty, S_lin, shading="auto", cmap=cmap_obj, norm=norm_lin)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("Spatial frequency (cpd)")
    ax0.set_ylabel("Temporal frequency (Hz)")
    ax0.set_title("CSF sensitivity (heatmap)")
    ax0.set_xticks(s_ticks)
    ax0.set_xticklabels([f"{v:g}" for v in s_ticks])
    ax0.set_yticks(t_ticks)
    ax0.set_yticklabels([f"{v:g}" for v in t_ticks])
    ax0.grid(True, which="both", ls="--", alpha=0.3)

    # 外置 colorbar（不会挤到图里）
    cb0 = fig.colorbar(pcm, ax=ax0, fraction=0.035, pad=0.01)
    cb0.set_label("Sensitivity")

    # ——— 右：3D 曲面（Z 轴 log 视觉；刻度/色条线性） ———
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax1.view_init(elev=30, azim=100)

    X_log = np.log10(Sx)
    Y_log = np.log10(Ty)
    facecolors = cmap_obj(norm_lin(S_lin))

    ax1.plot_surface(
        X_log,
        Y_log,
        Z_log,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        facecolors=facecolors,
    )
    ax1.set_title("CSF sensitivity (3D surface)")

    # X/Y：log 视觉但显示原值
    def _set_log_ticks(ax, axis, values):
        ticks = np.log10(values)
        labels = [f"{v:g}" for v in values]
        if axis == "x":
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

    _set_log_ticks(ax1, "x", s_ticks)
    _set_log_ticks(ax1, "y", t_ticks)
    ax1.set_xlim(np.log10(s_min), np.log10(s_max))
    ax1.set_ylim(np.log10(t_min), np.log10(t_max))

    # Z：log 视觉，线性标签
    def _linear_1_2_5_ticks(vmin, vmax):
        vmin = max(vmin, tiny)
        if vmax <= vmin:
            return [vmin]
        k_min = int(np.floor(np.log10(vmin)))
        k_max = int(np.ceil(np.log10(vmax)))
        bases = [1, 2, 5]
        ticks = []
        for k in range(k_min, k_max + 1):
            for b in bases:
                val = b * (10**k)
                if vmin <= val <= vmax:
                    ticks.append(val)
        return ticks or [vmin, vmax]

    z_lin_ticks = _linear_1_2_5_ticks(np.nanmin(S_pos), S_max)
    z_log_ticks = np.log10(z_lin_ticks)
    ax1.set_zticks(z_log_ticks)
    ax1.set_zticklabels([f"{v:g}" for v in z_lin_ticks])
    ax1.set_zlim(np.min(z_log_ticks), np.max(z_log_ticks))
    ax1.set_xlabel("Spatial frequency (cpd)")
    ax1.set_ylabel("Temporal frequency (Hz)")
    ax1.set_zlabel("Sensitivity")

    # 右图外置 colorbar（线性）
    import matplotlib as mpl  # 局部导入以保持上方更整洁

    mappable = mpl.cm.ScalarMappable(norm=norm_lin, cmap=cmap_obj)
    mappable.set_array([])
    cb1 = fig.colorbar(mappable, ax=ax1, fraction=0.035, pad=0.01)
    cb1.set_label("Sensitivity")

    # 峰值信息
    peak_idx = np.unravel_index(np.nanargmax(S_lin), S_lin.shape)
    peak_s = s_freqs[peak_idx[1]]
    peak_t = t_freqs[peak_idx[0]]
    peak_v = S_lin[peak_idx]
    print(f"Global peak sensitivity = {peak_v:.5g} at ~{peak_s:.4g} cpd, ~{peak_t:.4g} Hz")

    plt.show()
    return s_freqs, t_freqs, S


if __name__ == "__main__":
    model = CSF_CastleCSF()
    CSF_spatial_temporal_plot(
        csf_model=model,
        s_range=(0.1, 64.0),
        t_range=(0.1, 64.0),
        n_s=100,
        n_t=100,
        luminance=10.0,
        area=10.0,
        eccentricity=0.0,
        orientation=0.0,
        levels=15,
        figsize=(10, 4.5),
        cmap="viridis",
    )

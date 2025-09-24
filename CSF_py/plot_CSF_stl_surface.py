import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as cmaps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 如果这几个类不在同一文件，请按你的工程结构正确导入
from CSF_base import CSFBase
from CSF_castleCSF import CSF_CastleCSF
# 可选：如果你要替换为其他 CSF 模型
# from CSF_stelaCSF_lum_peak import CSF_StelaCSF_Lum_Peak
# from CSF_castleCSF_chrom import CSF_CastleCSF_Chrom

BASE_TICKS = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64]


def _pow10_ticks(vmin, vmax):
    """生成 luminance 轴 10 的整数次幂刻度"""
    vmin = max(float(vmin), 1e-12)
    k_min = int(np.floor(np.log10(vmin)))
    k_max = int(np.ceil(np.log10(vmax)))
    ticks = [10.0 ** k for k in range(k_min, k_max + 1)]
    return [t for t in ticks if vmin <= t <= vmax] or [vmin, vmax]


def _compute_slice_s_t(csf_model, s_vals, t_vals, fixed_L, area, ecc, ori):
    """返回形状 (n_t, n_s)，行对应 t，列对应 s"""
    n_s, n_t = len(s_vals), len(t_vals)
    S = np.empty((n_t, n_s), dtype=float)
    for i, tf in enumerate(t_vals):
        pars = {
            "s_frequency": s_vals,          # 向量
            "t_frequency": float(tf),       # 标量
            "orientation": ori,
            "luminance": float(fixed_L),
            "area": float(area),
            "eccentricity": float(ecc),
        }
        S[i, :] = csf_model.sensitivity(pars)
    return S


def _compute_slice_s_l(csf_model, s_vals, l_vals, fixed_t, area, ecc, ori):
    """返回形状 (n_l, n_s)，行对应 l，列对应 s"""
    n_s, n_l = len(s_vals), len(l_vals)
    S = np.empty((n_l, n_s), dtype=float)
    for i, Lv in enumerate(l_vals):
        pars = {
            "s_frequency": s_vals,          # 向量
            "t_frequency": float(fixed_t),  # 标量
            "orientation": ori,
            "luminance": float(Lv),         # 标量
            "area": float(area),
            "eccentricity": float(ecc),
        }
        S[i, :] = csf_model.sensitivity(pars)
    return S


def _compute_slice_t_l(csf_model, t_vals, l_vals, fixed_s, area, ecc, ori):
    """返回形状 (n_l, n_t)，行对应 l，列对应 t"""
    n_t, n_l = len(t_vals), len(l_vals)
    S = np.empty((n_l, n_t), dtype=float)
    for i, Lv in enumerate(l_vals):
        pars = {
            "s_frequency": float(fixed_s),  # 标量
            "t_frequency": t_vals,          # 向量
            "orientation": ori,
            "luminance": float(Lv),         # 标量
            "area": float(area),
            "eccentricity": float(ecc),
        }
        S[i, :] = csf_model.sensitivity(pars)
    return S


def _unify_norm_and_facecolors(Z_list, cmap_obj):
    """多图统一线性色标，返回 norm 与各图 facecolors"""
    S_max = 0.0
    for Z in Z_list:
        mask = np.isfinite(Z)
        if np.any(mask):
            S_max = max(S_max, float(np.nanmax(Z[mask])))
    S_max = max(S_max, 1.0)  # 防止全 0 或无穷
    norm_lin = mcolors.Normalize(vmin=0.0, vmax=S_max)
    faces = []
    for Z in Z_list:
        Zc = np.array(Z, dtype=float)
        Zc[~np.isfinite(Zc)] = 0.0
        Zc[Zc < 0] = 0.0
        faces.append(cmap_obj(norm_lin(Zc)))
    return norm_lin, faces


def CSF_spatial_temporal_luminance_plots(
    csf_model: CSFBase | None = None,
    # 频率/亮度取值范围（对数均匀）
    s_range: tuple[float, float] = (0.1, 64.0),
    t_range: tuple[float, float] = (0.1, 64.0),
    l_range: tuple[float, float] = (0.01, 10000.0),
    n_s: int = 120,
    n_t: int = 120,
    n_l: int = 120,
    # 每行固定的第三维默认值
    st_luminance: float = 100.0,   # s–t 切片固定的 L
    sl_fixed_t: float = 2.0,       # s–l 切片固定的 t
    tl_fixed_s: float = 2.0,       # t–l 切片固定的 s
    # 其它刺激参数
    area: float = 100.0,
    eccentricity: float = 0.0,
    orientation: float = 0.0,
    # 画图参数
    figsize: tuple[float, float] = (9, 12),
    cmap: str = "viridis",
):
    """
    三行两列：左热力图 + 右 3D 曲面（Z=log10(sensitivity)，色标为线性敏感度）
    返回：
        dict {
          "s_vals","t_vals","l_vals",
          "sens_st"(n_t,n_s),
          "sens_sl"(n_l,n_s),
          "sens_tl"(n_l,n_t)
        }
    """
    if csf_model is None:
        csf_model = CSF_CastleCSF()

    # 对数采样
    s_min, s_max = s_range
    t_min, t_max = t_range
    l_min, l_max = l_range
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), n_s)
    t_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    l_vals = np.logspace(np.log10(l_min), np.log10(l_max), n_l)

    # 计算三种切片
    sens_st = _compute_slice_s_t(csf_model, s_vals, t_vals, st_luminance, area, eccentricity, orientation)
    sens_sl = _compute_slice_s_l(csf_model, s_vals, l_vals, sl_fixed_t, area, eccentricity, orientation)
    sens_tl = _compute_slice_t_l(csf_model, t_vals, l_vals, tl_fixed_s, area, eccentricity, orientation)

    # 统一颜色归一化
    cmap_obj = cmaps.get_cmap(cmap)
    norm_lin, (fc_st, fc_sl, fc_tl) = _unify_norm_and_facecolors([sens_st, sens_sl, sens_tl], cmap_obj)

    # 画图：3 行 2 列
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        3, 2,
        left=0.11, right=0.93, bottom=0.05, top=0.97,
        wspace=0.01, hspace=0.06
    )

    # 预先准备一些刻度
    s_ticks = [v for v in BASE_TICKS if s_min <= v <= s_max]
    t_ticks = [v for v in BASE_TICKS if t_min <= v <= t_max]
    l_ticks = _pow10_ticks(l_min, l_max)

    #——— Row 1: s–t @ L=const ———#
    # 左：热力图
    ax11 = fig.add_subplot(gs[0, 0])
    S_st, T_st = np.meshgrid(s_vals, t_vals, indexing="xy")
    im11 = ax11.pcolormesh(S_st, T_st, np.clip(sens_st, 0, None), shading="auto",
                           cmap=cmap_obj, norm=norm_lin)
    ax11.set_xscale("log"); ax11.set_yscale("log")
    ax11.set_xlim(s_min, s_max); ax11.set_ylim(t_min, t_max)
    ax11.set_xticks(s_ticks); ax11.set_xticklabels([f"{v:g}" for v in s_ticks])
    ax11.set_yticks(t_ticks); ax11.set_yticklabels([f"{v:g}" for v in t_ticks])
    ax11.grid(True, which="both", ls="--", alpha=0.3)
    ax11.set_xlabel("Spatial frequency (cpd)")
    ax11.set_ylabel("Temporal frequency (Hz)")
    ax11.set_title(f"s–t @ L={st_luminance:g} cd/m² (heatmap)")

    # 右：3D 曲面
    ax12 = fig.add_subplot(gs[0, 1], projection="3d")
    ax12.view_init(elev=30, azim=100)
    X_log = np.log10(S_st); Y_log = np.log10(T_st)
    Z_log = np.log10(np.clip(sens_st, 1e-12, None))
    _ = ax12.plot_surface(X_log, Y_log, Z_log, rstride=1, cstride=1,
                          linewidth=0, antialiased=True, facecolors=fc_st)
    ax12.set_xticks(np.log10(s_ticks)); ax12.set_xticklabels([f"{v:g}" for v in s_ticks])
    ax12.set_yticks(np.log10(t_ticks)); ax12.set_yticklabels([f"{v:g}" for v in t_ticks])
    ax12.set_xlim(np.log10(s_min), np.log10(s_max))
    ax12.set_ylim(np.log10(t_min), np.log10(t_max))
    # Z 轴线性刻度标签（显示用）
    z_label_ticks_lin = [0.1, 1, 10, 100, 1000]
    z_ticks_log = np.log10(z_label_ticks_lin)
    ax12.set_zticks(z_ticks_log); ax12.set_zticklabels([f"{v:g}" for v in z_label_ticks_lin])
    ax12.set_zlim(np.log10(0.1), np.log10(1000))
    ax12.set_xlabel("Spatial frequency (cpd)")
    ax12.set_ylabel("Temporal frequency (Hz)")
    ax12.set_zlabel("Sensitivity")
    ax12.set_title(f"s–t @ L={st_luminance:g} cd/m² (3D)")

    #——— Row 2: s–l @ t=const ———#
    ax21 = fig.add_subplot(gs[1, 0])
    S_sl, L_sl = np.meshgrid(s_vals, l_vals, indexing="xy")
    im21 = ax21.pcolormesh(S_sl, L_sl, np.clip(sens_sl, 0, None), shading="auto",
                           cmap=cmap_obj, norm=norm_lin)
    ax21.set_xscale("log"); ax21.set_yscale("log")
    ax21.set_xlim(s_min, s_max); ax21.set_ylim(l_min, l_max)
    ax21.set_xticks(s_ticks); ax21.set_xticklabels([f"{v:g}" for v in s_ticks])
    l_ticks2 = _pow10_ticks(l_min, l_max)
    ax21.set_yticks(l_ticks2); ax21.set_yticklabels([f"{v:g}" for v in l_ticks2])
    ax21.grid(True, which="both", ls="--", alpha=0.3)
    ax21.set_xlabel("Spatial frequency (cpd)")
    ax21.set_ylabel("Luminance (cd/m²)")
    ax21.set_title(f"s–l @ tf={sl_fixed_t:g} Hz (heatmap)")

    ax22 = fig.add_subplot(gs[1, 1], projection="3d")
    ax22.view_init(elev=30, azim=100)
    X_log = np.log10(S_sl); Y_log = np.log10(L_sl)
    Z_log = np.log10(np.clip(sens_sl, 1e-12, None))
    _ = ax22.plot_surface(X_log, Y_log, Z_log, rstride=1, cstride=1,
                          linewidth=0, antialiased=True, facecolors=fc_sl)
    ax22.set_xticks(np.log10(s_ticks)); ax22.set_xticklabels([f"{v:g}" for v in s_ticks])
    ax22.set_yticks(np.log10(l_ticks2)); ax22.set_yticklabels([f"{v:g}" for v in l_ticks2])
    ax22.set_xlim(np.log10(s_min), np.log10(s_max))
    ax22.set_ylim(np.log10(l_min), np.log10(l_max))
    ax22.set_zticks(z_ticks_log); ax22.set_zticklabels([f"{v:g}" for v in z_label_ticks_lin])
    ax22.set_zlim(np.log10(0.1), np.log10(1000))
    ax22.set_xlabel("Spatial frequency (cpd)")
    ax22.set_ylabel("Luminance (cd/m²)")
    ax22.set_zlabel("Sensitivity")
    ax22.set_title(f"s–l @ tf={sl_fixed_t:g} Hz (3D)")

    #——— Row 3: t–l @ s=const ———#
    ax31 = fig.add_subplot(gs[2, 0])
    T_tl, L_tl = np.meshgrid(t_vals, l_vals, indexing="xy")
    im31 = ax31.pcolormesh(T_tl, L_tl, np.clip(sens_tl, 0, None), shading="auto",
                           cmap=cmap_obj, norm=norm_lin)
    ax31.set_xscale("log"); ax31.set_yscale("log")
    ax31.set_xlim(t_min, t_max); ax31.set_ylim(l_min, l_max)
    ax31.set_xticks(t_ticks); ax31.set_xticklabels([f"{v:g}" for v in t_ticks])
    l_ticks3 = _pow10_ticks(l_min, l_max)
    ax31.set_yticks(l_ticks3); ax31.set_yticklabels([f"{v:g}" for v in l_ticks3])
    ax31.grid(True, which="both", ls="--", alpha=0.3)
    ax31.set_xlabel("Temporal frequency (Hz)")
    ax31.set_ylabel("Luminance (cd/m²)")
    ax31.set_title(f"t–l @ sf={tl_fixed_s:g} cpd (heatmap)")

    ax32 = fig.add_subplot(gs[2, 1], projection="3d")
    ax32.view_init(elev=30, azim=100)
    X_log = np.log10(T_tl); Y_log = np.log10(L_tl)
    Z_log = np.log10(np.clip(sens_tl, 1e-12, None))
    _ = ax32.plot_surface(X_log, Y_log, Z_log, rstride=1, cstride=1,
                          linewidth=0, antialiased=True, facecolors=fc_tl)
    ax32.set_xticks(np.log10(t_ticks)); ax32.set_xticklabels([f"{v:g}" for v in t_ticks])
    ax32.set_yticks(np.log10(l_ticks3)); ax32.set_yticklabels([f"{v:g}" for v in l_ticks3])
    ax32.set_xlim(np.log10(t_min), np.log10(t_max))
    ax32.set_ylim(np.log10(l_min), np.log10(l_max))
    ax32.set_zticks(z_ticks_log); ax32.set_zticklabels([f"{v:g}" for v in z_label_ticks_lin])
    ax32.set_zlim(np.log10(0.1), np.log10(1000))
    ax32.set_xlabel("Temporal frequency (Hz)")
    ax32.set_ylabel("Luminance (cd/m²)")
    ax32.set_zlabel("Sensitivity")
    ax32.set_title(f"t–l @ sf={tl_fixed_s:g} cpd (3D)")

    # 全局 colorbar（与所有子图共享 norm + cmap）
    import matplotlib as mpl
    mappable_global = mpl.cm.ScalarMappable(norm=norm_lin, cmap=cmap_obj)
    mappable_global.set_array([])
    cb_global = fig.colorbar(
        mappable_global,
        ax=[ax11, ax12, ax21, ax22, ax31, ax32],
        location="right",
        fraction=0.035,
        pad=0.05
    )
    cb_global.set_label("Sensitivity (1/threshold)")

    # 峰值信息（线性敏感度）
    def _peak_info(Z, X_vals, Y_vals, xname, yname):
        Zp = np.clip(np.array(Z, dtype=float), 0, None)
        if not np.isfinite(Zp).any():
            print(f"[WARN] {xname}-{yname} 切片全为非有限值")
            return
        idx = np.unravel_index(np.nanargmax(Zp), Zp.shape)
        xv = X_vals[idx[1]]
        yv = Y_vals[idx[0]]
        mv = Zp[idx]
        print(f"Peak (linear sensitivity) on {xname}-{yname}: {mv:.5g} at {xname}~{xv:.4g}, {yname}~{yv:.4g}")

    _peak_info(sens_st, s_vals, t_vals, "s", "t")
    _peak_info(sens_sl, s_vals, l_vals, "s", "l")
    _peak_info(sens_tl, t_vals, l_vals, "t", "l")

    plt.show()

    return {
        "s_vals": s_vals, "t_vals": t_vals, "l_vals": l_vals,
        "sens_st": sens_st, "sens_sl": sens_sl, "sens_tl": sens_tl,
    }


if __name__ == "__main__":
    model = CSF_CastleCSF()
    CSF_spatial_temporal_luminance_plots(
        csf_model=model,
        s_range=(0.1, 64.0),
        t_range=(0.1, 64.0),
        l_range=(0.01, 10000.0),
        n_s=120, n_t=120, n_l=120,
        st_luminance=100.0,   # 第 1 行固定的 L
        sl_fixed_t=2.0,       # 第 2 行固定的 t
        tl_fixed_s=2.0,       # 第 3 行固定的 s
        area=100.0,
        eccentricity=0.0,
        orientation=0.0,
        figsize=(9, 12),
        cmap="viridis",
    )

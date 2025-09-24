import numpy as np
import matplotlib.pyplot as plt

# 如果这两个类不在同一文件，请按你的工程结构正确导入
from CSF_base import CSFBase
from CSF_stelaCSF_lum_peak import CSF_StelaCSF_Lum_Peak
from CSF_castleCSF_chrom import CSF_CastleCSF_Chrom
from CSF_castleCSF import CSF_CastleCSF
# === 1) 初始化模型 ===
# csf_model = CSF_StelaCSF_Lum_Peak()
# csf_model = CSF_CastleCSF_Chrom('rg')
# csf_model = CSF_CastleCSF_Chrom('yv')
csf_model = CSF_CastleCSF()

# === 2) 构造参数（仿照 MATLAB 示例）===
# s_freqs = logspace(log10(0.1), log10(64), 100);
s_freqs = np.logspace(np.log10(0.1), np.log10(64.0), 100)

csf_pars = {
    "s_frequency": s_freqs,   # 空间频率 (cpd)
    "t_frequency": 10.0,      # 时间频率 (Hz)
    "orientation": 0.0,       # 方向 (deg) - 当前模型未使用
    "luminance": 100.0,       # 亮度 (cd/m^2)
    "area": 100.0,            # 刺激面积 (deg^2)
    "eccentricity": 0.0,      # 离心率 (deg)
    # "vis_field": 180.0,     # 可选；不填则基类会给默认值 180
}

# === 3) 计算敏感度 ===
sensitivities = csf_model.sensitivity(csf_pars)

# === 4) 画图（等价 MATLAB 的 semilogx）===
plt.figure()
plt.semilogx(s_freqs, sensitivities, linewidth=2)
plt.grid(True, which="both")
plt.xlabel("Spatial frequency (cpd)")
plt.ylabel("Sensitivity")
plt.title("Contrast Sensitivity Function (stelaCSF_lum_peak)")
plt.show()

# === 5)（可选）打印一些数值检查 ===
print(f"min S = {np.min(sensitivities):.5g}, max S = {np.max(sensitivities):.5g}")
print(f"peak at ~{s_freqs[np.argmax(sensitivities)]:.4g} cpd")

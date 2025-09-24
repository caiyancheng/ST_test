import numpy as np
import torch
from typing import Union, Any

ArrayLike = Union[float, int, np.ndarray, torch.Tensor]

class display_encode:
    def __init__(self, display_encoded_a: float, Y_black=0, Y_refl=0):
        self.display_encoded_a = float(display_encoded_a)
        self.Y_black = Y_black  # 0.2
        self.Y_refl = Y_refl    # 0.3978873577297384

    # ---------- helpers ----------
    @staticmethod
    def _is_tensor(x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    @staticmethod
    def _is_np(x: Any) -> bool:
        return isinstance(x, np.ndarray) or np.isscalar(x)

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        return np.asarray(x)

    @staticmethod
    def _to_tensor(x: ArrayLike, like: torch.Tensor = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=like.dtype if like is not None else None,
                            device=like.device if like is not None else None)

    @staticmethod
    def _restore_numpy_type(x_in: ArrayLike, x_out_np: np.ndarray) -> ArrayLike:
        # 标量保持标量
        if np.isscalar(x_in):
            return float(x_out_np)
        return x_out_np

    @staticmethod
    def _restore_tensor_type(x_in: ArrayLike, x_out_t: torch.Tensor) -> ArrayLike:
        if isinstance(x_in, torch.Tensor):
            return x_out_t
        # 非 tensor 输入时，返回 python 标量或 numpy（标量→标量，其他→numpy）
        if x_out_t.numel() == 1:
            return float(x_out_t.item())
        return x_out_t.detach().cpu().numpy()

    # ---------- gamma ----------
    def L2C_gamma(self, Luminance: ArrayLike) -> ArrayLike:
        a = self.display_encoded_a
        if self._is_tensor(Luminance):
            L = self._to_tensor(Luminance)
            C = (L / a) ** (1.0 / 2.2)
            return self._restore_tensor_type(Luminance, C)
        else:
            L = self._to_numpy(Luminance)
            C = (L / a) ** (1.0 / 2.2)
            return self._restore_numpy_type(Luminance, C)

    def C2L_gamma(self, Color: ArrayLike) -> ArrayLike:
        a = self.display_encoded_a
        if self._is_tensor(Color):
            C = self._to_tensor(Color)
            L = a * (C ** 2.2)
            return self._restore_tensor_type(Color, L)
        else:
            C = self._to_numpy(Color)
            L = a * (C ** 2.2)
            return self._restore_numpy_type(Color, L)

    # ---------- sRGB ----------
    # 通用 srgb -> linear（相对亮度）转换；支持 torch / numpy / 标量
    @staticmethod
    def srgb2lin(p: ArrayLike) -> ArrayLike:
        # torch 分支
        if isinstance(p, torch.Tensor):
            return torch.where(p > 0.04045, ((p + 0.055) / 1.055) ** 2.4, p / 12.92)
        # numpy / 标量 分支
        p_np = np.asarray(p)
        out = np.where(p_np > 0.04045, ((p_np + 0.055) / 1.055) ** 2.4, p_np / 12.92)
        if np.isscalar(p):
            return float(out)
        return out

    # L2C_sRGB: 输入是亮度（同单位），输出是编码 [0,1]
    def L2C_sRGB(self, Luminance: ArrayLike) -> ArrayLike:
        a = self.display_encoded_a
        thr = a * 0.04045 / 12.92  # 亮度域阈值

        if self._is_tensor(Luminance):
            L = self._to_tensor(Luminance)
            low = (L * 12.92) / a
            high = 1.055 * torch.pow(L / a, 1.0 / 2.4) - 0.055
            C = torch.where(L <= thr, low, high)
            return self._restore_tensor_type(Luminance, C)
        else:
            L = self._to_numpy(Luminance)
            low = (L * 12.92) / a
            high = 1.055 * np.power(L / a, 1.0 / 2.4) - 0.055
            C = np.where(L <= thr, low, high)
            return self._restore_numpy_type(Luminance, C)

    # C2L_sRGB: 输入是 sRGB 编码 [0,1]，输出是亮度
    # —— 改为直接用 srgb2lin
    def C2L_sRGB(self, Color: ArrayLike) -> ArrayLike:
        a = self.display_encoded_a
        lin = self.srgb2lin(Color)
        if isinstance(lin, torch.Tensor):
            L = a * lin
            return self._restore_tensor_type(Color, L)
        else:
            L = a * np.asarray(lin)
            return self._restore_numpy_type(Color, L)

    # 带黑位/反射项的版本，同样复用 srgb2lin
    def C2L_sRGB_display(self, Color: ArrayLike) -> ArrayLike:
        a = self.display_encoded_a
        lin = self.srgb2lin(Color)
        if isinstance(lin, torch.Tensor):
            L = (a - self.Y_black) * lin + self.Y_refl + self.Y_black
            return self._restore_tensor_type(Color, L)
        else:
            L = (a - self.Y_black) * np.asarray(lin) + self.Y_refl + self.Y_black
            return self._restore_numpy_type(Color, L)


if __name__ == "__main__":
    enc = display_encode(1000)
    Color = enc.L2C_sRGB(400)  # 亮度->sRGB
    print(Color)
    # 反向（使用 srgb2lin）：
    L = enc.C2L_sRGB(Color)
    print(L)

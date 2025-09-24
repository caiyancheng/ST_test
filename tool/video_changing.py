from typing import Union

try:
    import torch
    _has_torch = True
except Exception:
    torch = None
    _has_torch = False

ArrayLike = Union["torch.Tensor", "numpy.ndarray"]  # 仅作类型提示，运行时不强依赖


def check_target_fps(target_fps: int, src_fps: int = 240) -> int:
    """
    检查 target_fps 是否为 src_fps 的约数，并返回步长 step = src_fps // target_fps
    """
    if not isinstance(target_fps, int):
        raise TypeError("TARGET_FPS 必须为正整数。")
    if target_fps <= 0:
        raise ValueError("TARGET_FPS 必须为正整数。")
    if src_fps % target_fps != 0:
        raise ValueError(f"TARGET_FPS={target_fps} 不是 {src_fps} 的公约数。")
    return src_fps // target_fps  # 步长


def _is_torch_tensor(x) -> bool:
    return _has_torch and isinstance(x, torch.Tensor)


def downsample_decimate(video: ArrayLike, step: int) -> ArrayLike:
    """
    仅抽帧，不做插值或平滑。
    输入/输出均为 [B, C, F, H, W]，沿 F 维以步长 step 下采样。

    参数:
        video: numpy.ndarray 或 torch.Tensor，形状 [B, C, F, H, W]
        step:  下采样步长 = SRC_FPS // TARGET_FPS
    返回:
        与输入类型相同的数组/张量，形状 [B, C, F', H, W]，其中 F' = ceil(F / step)
    """
    if video.ndim != 5:
        raise ValueError(f"期望 5D [B, C, F, H, W]，但得到 {tuple(video.shape)}")
    # 直接沿维度 2（F 维）做步进切片
    return video[:, :, ::step, :, :]


def frame_hold_like_reference(video: ArrayLike, step: int) -> ArrayLike:
    """
    “首帧保持”（Frame Hold）机制：
    将时间轴按长度为 step 的块划分，每个块内所有帧都替换为该块的首帧。
    不做插值或平滑。

    输入/输出:
        形状均为 [B, C, F, H, W]；返回的 F 与输入一致。
    """
    if video.ndim != 5:
        raise ValueError(f"期望 5D [B, C, F, H, W]，但得到 {tuple(video.shape)}")

    F = video.shape[2]
    if F <= 0:
        return video

    if _is_torch_tensor(video):
        # 在同一设备上构造索引
        idx_seq = torch.arange(F, device=video.device) // step * step
        idx_seq = idx_seq.to(dtype=torch.long)
        # 沿时间维度（dim=2）索引复制
        out = video.index_select(dim=2, index=idx_seq)
        return out
    else:
        import numpy as np
        idx_seq = (np.arange(F) // step) * step
        # 直接在轴 2 做高级索引
        out = video[:, :, idx_seq, :, :]
        return out


# -----------------------------
# 简短用例（伪代码，仅示意）
# -----------------------------
# np_input: np.ndarray, shape [B, C, F, H, W]
# torch_input: torch.Tensor, shape [B, C, F, H, W]

# step = check_target_fps(target_fps=60, src_fps=240)  # step = 4
# np_ds   = downsample_decimate(np_input, step)        # [B, C, F/4, H, W]
# np_hold = frame_hold_like_reference(np_input, step)  # [B, C, F,   H, W]
# t_ds    = downsample_decimate(torch_input, step)     # [B, C, F/4, H, W]
# t_hold  = frame_hold_like_reference(torch_input, step) # [B, C, F, H, W]

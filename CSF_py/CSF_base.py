from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Iterable, Union, IO

ArrayLike = Union[np.ndarray, float, int]

class CSFBase(ABC):
    """
    Spatio-chromatic CSF 模型的基类（Python 版）
    - par: 参数（嵌套 dict / 标量 / 一维向量）
    - mean_error / mean_std: 训练或评估时可用
    """

    def __init__(self):
        self.par: Dict[str, Any] = {}
        self.mean_error: float | None = None
        self.mean_std: float | None = None

    # ------- 抽象接口 -------
    @abstractmethod
    def short_name(self) -> str:
        """返回简短名称（可用于文件名等）"""
        raise NotImplementedError

    def full_name(self) -> str:
        """可被子类覆盖；默认等于 short_name"""
        return self.short_name()

    @abstractmethod
    def sensitivity(self, pars: Dict[str, Any], *args, **kwargs) -> np.ndarray:
        """
        统一的敏感度计算接口。
        参照 MATLAB 注释：支持 luminance/lms_bkg, s_frequency, t_frequency, orientation,
        lms_delta, area/ge_sigma, eccentricity, vis_field 等字段；
        字段尺寸须可广播（broadcastable）。
        """
        raise NotImplementedError

    # ------- 训练辅助：参数向量 <-> 结构 -------
    def set_pars(self, pars_vector: np.ndarray):
        """按给定向量更新 self.par（用于参数优化）。"""
        assert self.par is not None and self.par != {}, "self.par 必须先初始化（例如 get_default_par）"
        self.par, _ = self.param2struct(self.par, np.asarray(pars_vector).ravel())

    def get_pars(self) -> np.ndarray:
        """把 self.par 展平成向量（用于参数优化）。"""
        return self.struct2param(self.par)

    # ------- 便捷包装（与 MATLAB 名称保持一致）-------
    def sensitivity_stolms(
        self, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity
    ):
        csf_pars = dict(
            s_frequency=s_freq,
            t_frequency=t_freq,
            orientation=orientation,
            lms_bkg=LMS_bkg,
            lms_delta=LMS_delta,
            area=area,
            eccentricity=eccentricity,
        )
        return self.sensitivity(csf_pars)

    def sensitivity_stolms_jov(
        self, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity, col_dir
    ):
        # 基类不处理 col_dir；留给特定子类覆写
        return self.sensitivity_stolms(
            s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity
        )

    def sensitivity_stolmsv(
        self, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity, vis_field
    ):
        csf_pars = dict(
            s_frequency=s_freq,
            t_frequency=t_freq,
            orientation=orientation,
            lms_bkg=LMS_bkg,
            lms_delta=LMS_delta,
            area=area,
            eccentricity=eccentricity,
            vis_field=vis_field,
        )
        return self.sensitivity(csf_pars)

    def sensitivity_stolmsv_jov(
        self,
        s_freq,
        t_freq,
        orientation,
        LMS_bkg,
        LMS_delta,
        area,
        eccentricity,
        vis_field,
        col_dir,
    ):
        # 基类不处理 col_dir；留给特定子类覆写
        return self.sensitivity_stolmsv(
            s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity, vis_field
        )

    def sensitivity_stolms_edge(
        self, t_freq, orientation, LMS_bkg, LMS_delta, ge_sigma, eccentricity
    ):
        csf_pars = dict(
            t_frequency=t_freq,
            orientation=orientation,
            lms_bkg=LMS_bkg,
            lms_delta=LMS_delta,
            ge_sigma=ge_sigma,
            eccentricity=eccentricity,
        )
        # 具体 edge 版本由子类实现（如有）
        if not hasattr(self, "sensitivity_edge"):
            raise NotImplementedError("该模型未实现 sensitivity_edge()")
        return getattr(self, "sensitivity_edge")(csf_pars)

    def sensitivity_stolms_edge_jov(
        self, t_freq, orientation, LMS_bkg, LMS_delta, ge_sigma, eccentricity, col_dir
    ):
        # 基类不处理 col_dir；留给特定子类覆写
        return self.sensitivity_stolms_edge(
            t_freq, orientation, LMS_bkg, LMS_delta, ge_sigma, eccentricity
        )

    # ------- 参数校验与补全 -------
    def test_complete_params(
            self,
            pars: Dict[str, Any],
            requires: Iterable[str] = (),
            expand: bool = False,  # 保持占位
    ) -> Dict[str, Any]:
        """
        - 名称校验
        - 广播尺寸校验：对 lms_* 仅用其去掉最后一维的形状检查；末维必须为 3
        - ‘互斥但必有其一’ 的自动补齐（luminance<->lms_bkg, area<->ge_sigma）
        - 默认参数补齐
        """
        valid_names = {
            "luminance",
            "lms_bkg",
            "lms_delta",
            "s_frequency",
            "t_frequency",
            "orientation",
            "area",
            "ge_sigma",
            "eccentricity",
            "vis_field",
        }

        pars = dict(pars)  # 复制，避免原地改
        fn = list(pars.keys())

        # 用“形状推导”而不是数值相乘做广播检查
        cur_shape: tuple = ()
        color_ndim: int | None = None

        def _broadcast_shapes(a: tuple, b: tuple) -> tuple:
            # 兼容旧 numpy
            try:
                import numpy as _np
                return _np.broadcast_shapes(a, b)
            except Exception:
                a_arr = np.empty(a)
                b_arr = np.empty(b)
                return np.broadcast(a_arr, b_arr).shape

        for name in fn:
            if name not in valid_names:
                raise ValueError(f"参数结构包含未识别字段 '{name}'")

            param = np.asarray(pars[name])

            if name in ("lms_bkg", "lms_delta"):
                # 末维必须为 3；颜色维所在的“维度数”需一致
                if param.shape[-1] != 3:
                    raise ValueError(f"'{name}' 的最后一维必须为 3")
                if color_ndim is not None and color_ndim != param.ndim:
                    raise ValueError(
                        f"LMS 颜色必须出现在相同的维度（维度数不一致：{color_ndim} vs {param.ndim})"
                    )
                color_ndim = param.ndim

                # 仅用“去掉最后一维”的形状参与广播检查
                pshape_nc = param.shape[:-1]
                try:
                    cur_shape = _broadcast_shapes(cur_shape, pshape_nc)
                except Exception as e:
                    raise ValueError(f"参数 '{name}' 的尺寸无法广播（除去颜色维后）") from e
            else:
                try:
                    cur_shape = _broadcast_shapes(cur_shape, param.shape)
                except Exception as e:
                    raise ValueError(f"参数 '{name}' 的尺寸无法广播") from e

        if color_ndim is None:
            # 若未提供任何 lms_*，默认颜色维度数 = 非颜色公共维数 + 1
            color_ndim = len(cur_shape) + 1

        # ---- 互斥参数自动补齐 ----
        if "luminance" in requires:
            if "luminance" not in pars:
                if "lms_bkg" not in pars:
                    raise ValueError("需要提供 'luminance' 或 'lms_bkg' 之一")
                lm = np.asarray(pars["lms_bkg"])
                L = self.last_dim(lm, 1)
                M = self.last_dim(lm, 2)
                pars["luminance"] = L + M

        if "lms_bkg" in requires:
            if "lms_bkg" not in pars:
                if "luminance" not in pars:
                    raise ValueError("需要提供 'luminance' 或 'lms_bkg' 之一")
                d65 = np.array([0.6991, 0.3009, 0.0198], dtype=float)
                shape = (1,) * (color_ndim - 1) + (3,)
                pars["lms_bkg"] = d65.reshape(shape) * np.asarray(pars["luminance"])

        if "ge_sigma" in requires:
            if "ge_sigma" not in pars:
                if "area" not in pars:
                    raise ValueError("需要提供 'ge_sigma' 或 'area' 之一")
                pars["ge_sigma"] = np.sqrt(np.asarray(pars["area"]) / np.pi)

        if "area" in requires:
            if "area" not in pars:
                if "ge_sigma" not in pars:
                    raise ValueError("需要提供 'ge_sigma' 或 'area' 之一")
                pars["area"] = np.pi * (np.asarray(pars["ge_sigma"]) ** 2)

        # ---- 默认参数 ----
        def_lms_delta = np.array([0.6855, 0.2951, 0.0194], dtype=float).reshape(
            (1,) * (color_ndim - 1) + (3,)
        )
        def_pars = dict(
            eccentricity=0,
            vis_field=180,
            orientation=0,
            t_frequency=0,
            lms_delta=def_lms_delta,
        )
        for k, v in def_pars.items():
            if k not in pars:
                pars[k] = v

        return pars

    # ------- 输出参数（便于粘贴到 get_default_par） -------
    def print(self, fh: IO[str]):
        """把 self.par 以可复制格式打印到文件句柄 fh"""
        self.print_struct(fh, "p.", self.par)

    def print_struct(self, fh: IO[str], struct_name: str, s: Dict[str, Any]):
        """递归打印 dict（跳过特定字段名）"""
        skip = {"cm", "ds", "sust", "trans"}
        for k, v in s.items():
            if k in skip:
                continue
            if isinstance(v, dict):
                self.print_struct(fh, struct_name + k + ".", v)
            else:
                fh.write(f"\t{struct_name}{k} = ")
                self.print_vector(fh, np.asarray(v))
                fh.write(";\n")

    # ------- 静态工具（类方法实现以便子类也能用） -------
    @staticmethod
    def update_struct(src: Dict[str, Any], dst: Dict[str, Any]) -> Dict[str, Any]:
        """把 src 中已有键的值更新到 dst（尺寸需一致）；支持嵌套 dict。"""
        for k, v in src.items():
            if k in dst:
                if isinstance(dst[k], dict) and isinstance(v, dict):
                    dst[k] = CSFBase.update_struct(v, dst[k])
                else:
                    a = np.asarray(dst[k])
                    b = np.asarray(v)
                    if a.shape != b.shape:
                        raise AssertionError(
                            f"update_struct: 字段 '{k}' 尺寸不一致 {a.shape} vs {b.shape}"
                        )
                    dst[k] = v
        return dst

    @staticmethod
    def sel_dim(X: np.ndarray, d: int) -> np.ndarray:
        """
        MATLAB 版本的 sel_dim：
          cln(1:ndims(X))={1}; 若 d>1: cln(d)=':'；否则 cln(end-d+1)=':'
        直观理解：除某一维外，其他维都取索引 1（Python 用 0），该维取全切片。
        支持 d>0（从前往后计）或 d<=0（从后往前计，类似负索引）。
        """
        X = np.asarray(X)
        ndim = X.ndim
        if ndim == 0:
            return X  # 标量

        if d > 0:
            axis = d - 1
        else:
            axis = ndim + d  # 负数从后往前

        axis = max(0, min(ndim - 1, axis))
        index = [0] * ndim
        index[axis] = slice(None)
        return X[tuple(index)]

    @staticmethod
    def last_dim(X: np.ndarray, d: int) -> np.ndarray:
        """
        取“最后一维”的第 d 个元素（MATLAB 的 1-based 下标）。
        MATLAB:
          cln(:)=':'; cln{end}=d; Y=X(cln{:});
        Python 对应：axis=-1，index=d-1
        """
        X = np.asarray(X)
        return np.take(X, indices=d - 1, axis=-1)

    @staticmethod
    def param2struct(s: Dict[str, Any], pars_vector: np.ndarray) -> Tuple[Dict[str, Any], int]:
        """
        把参数向量写回到与 s 同构的 dict（深度优先，按键顺序）。
        返回 (更新后的 dict, 已消耗的元素个数)
        约定：叶子节点是标量或一维向量；向量长度决定消耗个数。
        """
        pars_vector = np.asarray(pars_vector).ravel()
        pos = 0
        out = {}

        for k, v in s.items():
            if isinstance(v, dict):
                sub, used = CSFBase.param2struct(v, pars_vector[pos:])
                out[k] = sub
                pos += used
            else:
                v = np.asarray(v)
                N = v.size
                out[k] = pars_vector[pos : pos + N].reshape(v.shape)
                pos += N

        return out, pos

    @staticmethod
    def struct2param(s: Dict[str, Any]) -> np.ndarray:
        """
        把嵌套 dict 展平成一维向量（深度优先，按键顺序）。
        """
        parts = []
        for k, v in s.items():
            if isinstance(v, dict):
                parts.append(CSFBase.struct2param(v))
            else:
                parts.append(np.asarray(v).ravel())
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts).astype(float)

    # ------- 亮度依赖函数族 -------
    @staticmethod
    def get_lum_dep(pars: Iterable[float], L: ArrayLike) -> np.ndarray:
        """
        同 MATLAB get_lum_dep：
          len=1: 常数
          len=2: v = pars[1]*L^pars[0]
          len=3: v = pars[0]*(1+pars[1]/L)^(-pars[2])
          len=5: v = pars[0]*(1+pars[1]/L)^(-pars[2]) * (1-(1+pars[3]/L)^(-pars[4]))
        """
        L = np.asarray(L, dtype=float)
        p = np.asarray(pars, dtype=float).ravel()
        n = p.size

        if n == 1:
            return np.ones_like(L) * p[0]
        if n == 2:
            return p[1] * (L ** p[0])
        if n == 3:
            return p[0] * (1.0 + p[1] / L) ** (-p[2])
        if n == 5:
            return p[0] * (1.0 + p[1] / L) ** (-p[2]) * (1.0 - (1.0 + p[3] / L) ** (-p[4]))
        raise NotImplementedError("get_lum_dep: 未实现的参数长度（只支持 1/2/3/5）")

    @staticmethod
    def get_lum_dep_dec(pars: Iterable[float], L: ArrayLike) -> np.ndarray:
        """
        同 MATLAB get_lum_dep_dec（随亮度递减的族）：
          len=1: 常数
          len=2: v = 10^(-p0*log10(L) + p1)
          len=3: v = p0 * (1 - (1 + p1/L)^(-p2))
        """
        L = np.asarray(L, dtype=float)
        log_lum = np.log10(L)
        p = np.asarray(pars, dtype=float).ravel()
        n = p.size

        if n == 1:
            return np.ones_like(L) * p[0]
        if n == 2:
            return 10.0 ** (-p[0] * log_lum + p[1])
        if n == 3:
            return p[0] * (1.0 - (1.0 + p[1] / L) ** (-p[2]))
        raise NotImplementedError("get_lum_dep_dec: 未实现的参数长度（只支持 1/2/3）")

    # ------- 数据集默认参数钩子 -------
    @staticmethod
    def get_dataset_par() -> Dict[str, Any]:
        """留给具体模型/数据集覆盖；默认返回空 dict。"""
        return {}

    # ------- 打印向量（便于粘贴） -------
    @staticmethod
    def print_vector(fh: IO[str], vec: np.ndarray):
        v = np.asarray(vec).ravel()
        if v.size > 1:
            fh.write("[ ")
            fh.write(" ".join(f"{x:g}" for x in v))
            fh.write(" ]")
        else:
            fh.write(f"{v.item():g}")

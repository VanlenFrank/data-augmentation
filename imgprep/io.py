"""图像读写模块."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

__all__ = ["imread", "imwrite"]


PathLike = Union[str, Path]


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图像文件.

    Args:
        path:  图像文件路径 (支持 .jpg/.png/.bmp/.tiff 等 OpenCV 支持的格式).
        flags: 读取模式，默认为 cv2.IMREAD_COLOR (BGR 三通道).
               常用值:
                - cv2.IMREAD_COLOR        : BGR 彩色 (默认)
                - cv2.IMREAD_GRAYSCALE    : 单通道灰度
                - cv2.IMREAD_UNCHANGED    : 保持原通道 (含 alpha)

    Returns:
        ndarray, shape=(H, W, C) 或 (H, W).

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError:        图像读取失败 (文件损坏或格式不支持).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    img = cv2.imread(str(path), flags)
    if img is None:
        raise ValueError(f"无法读取图像: {path} (文件可能已损坏或格式不支持)")

    return img


def imwrite(path: PathLike, img: np.ndarray) -> None:
    """保存图像到文件.

    格式由文件扩展名自动推断 (.jpg/.png/.bmp/.tiff 等).

    Args:
        path: 输出路径.
        img:  图像数据, ndarray.

    Raises:
        ValueError: 图像数据为空或保存失败.
        OSError:    目录不存在或权限不足.
    """
    path = Path(path)

    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.size == 0:
        raise ValueError("图像数据为空，无法保存.")

    success = cv2.imwrite(str(path), img)
    if not success:
        raise ValueError(f"保存图像失败: {path}")

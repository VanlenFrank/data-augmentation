"""噪声生成模块."""

import numpy as np

__all__ = ["salt_pepper_noise", "gaussian_noise"]


def salt_pepper_noise(
    img: np.ndarray,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01,
    salt_val: int = 255,
    pepper_val: int = 0,
) -> np.ndarray:
    """添加椒盐噪声.

    随机将图像中的部分像素置为纯白 (盐) 或纯黑 (椒).

    Args:
        img:         输入图像, uint8, ndarray.
        salt_prob:   盐噪声概率 (白点), 范围 [0, 1], 默认 0.01.
        pepper_prob: 椒噪声概率 (黑点), 范围 [0, 1], 默认 0.01.
        salt_val:    盐噪声像素值, 默认 255 (白色).
        pepper_val:  椒噪声像素值, 默认 0   (黑色).

    Returns:
        添加噪声后的图像, ndarray, dtype=uint8.

    Raises:
        ValueError: 概率超出 [0, 1) 范围.
    """
    if not 0 <= salt_prob < 1:
        raise ValueError(f"salt_prob 必须满足 0 <= prob < 1, 当前: {salt_prob}")
    if not 0 <= pepper_prob < 1:
        raise ValueError(f"pepper_prob 必须满足 0 <= prob < 1, 当前: {pepper_prob}")

    output = img.copy()
    total = img.size if img.ndim == 2 else img.shape[0] * img.shape[1]

    # 盐噪声: 随机位置置白
    if salt_prob > 0:
        num_salt = int(total * salt_prob)
        coords = np.random.choice(total, num_salt, replace=False)
        if img.ndim == 2:
            rows, cols = img.shape
            output.flat[coords] = salt_val
        else:
            rows, cols = img.shape[:2]
            r_idx = coords // cols
            c_idx = coords % cols
            output[r_idx, c_idx, :] = salt_val

    # 椒噪声: 随机位置置黑
    if pepper_prob > 0:
        num_pepper = int(total * pepper_prob)
        coords = np.random.choice(total, num_pepper, replace=False)
        if img.ndim == 2:
            output.flat[coords] = pepper_val
        else:
            r_idx = coords // cols
            c_idx = coords % cols
            output[r_idx, c_idx, :] = pepper_val

    return output


def gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
) -> np.ndarray:
    """添加高斯噪声.

    向图像添加服从 N(mean, sigma^2) 分布的高斯噪声.

    Args:
        img:   输入图像, uint8, ndarray.
        mean:  噪声均值,  默认 0.0.
        sigma: 噪声标准差, 默认 25.0 (值越大噪声越强).

    Returns:
        添加噪声后的图像, ndarray, dtype=uint8, 值域已裁剪到 [0, 255].

    Raises:
        ValueError: img 为空或 sigma < 0.
    """
    if img.size == 0:
        raise ValueError("输入图像为空.")
    if sigma < 0:
        raise ValueError(f"sigma 不能为负, 当前: {sigma}")

    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    output = img.astype(np.float32) + noise
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

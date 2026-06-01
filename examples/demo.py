"""imgprep 使用示例."""

import cv2
import numpy as np

from imgprep import (
    imread, imwrite,
    to_grayscale, to_hsv,
    adjust_brightness,
    salt_pepper_noise,
    motion_blur,
)


def main():
    # 生成一张测试图 (灰白渐变)
    width, height = 400, 300
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.tile(gradient, (height, 1, 1))
    img = np.repeat(img, 3, axis=2)  # (H, W, 3)
    print(f"生成测试图: shape={img.shape}, dtype={img.dtype}")

    # 1. 灰度转换
    gray = to_grayscale(img)
    print(f"灰度图: shape={gray.shape}")

    # 2. 通道转换
    hsv = to_hsv(img)
    print(f"HSV: shape={hsv.shape}")

    # 3. 亮度增强
    bright = adjust_brightness(img, alpha=1.2, beta=40)
    print(f"亮度增强: mean={bright.mean():.1f} (原始 mean={img.mean():.1f})")

    # 4. 椒盐噪声
    noisy = salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
    print(f"椒盐噪声: 共有 {int(0.04 * img.size / 3)} 个像素被污染")

    # 5. 运动模糊 (水平方向)
    blurred_h = motion_blur(img, kernel_size=21, angle=0)
    print(f"水平运动模糊: shape={blurred_h.shape}")

    # 6. 运动模糊 (45度方向)
    blurred_d = motion_blur(img, kernel_size=21, angle=45)
    print(f"45° 运动模糊: shape={blurred_d.shape}")

    # 保存结果
    results = {
        "original.jpg": img,
        "gray.jpg": gray,
        "bright.jpg": bright,
        "noisy.jpg": noisy,
        "blur_h.jpg": blurred_h,
        "blur_d.jpg": blurred_d,
    }
    for name, data in results.items():
        imwrite(f"output/{name}", data)
        print(f"  -> saved output/{name}")

    print("全部完成!")


if __name__ == "__main__":
    main()

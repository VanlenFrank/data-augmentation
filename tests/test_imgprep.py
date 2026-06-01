"""imgprep 单元测试."""

import numpy as np
import pytest

from imgprep.io import imread, imwrite
from imgprep.convert import (
    to_grayscale, to_rgb, to_bgr, to_hsv, to_lab, convert_color,
)
from imgprep.enhance import adjust_brightness
from imgprep.noise import salt_pepper_noise, gaussian_noise
from imgprep.blur import motion_blur


@pytest.fixture
def fake_bgr():
    """生成一张 100x200 的彩色测试图."""
    return np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def fake_gray():
    """生成一张 100x200 的灰度测试图."""
    return np.random.randint(0, 256, (100, 200), dtype=np.uint8)


# ============ io ============

def test_imread_not_found():
    with pytest.raises(FileNotFoundError):
        imread("/nonexistent/path.jpg")


def test_imwrite_empty(fake_bgr):
    empty = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError, match="为空"):
        imwrite("/tmp/_test_empty.jpg", empty)


# ============ convert ============

class TestConvert:
    def test_to_grayscale(self, fake_bgr):
        gray = to_grayscale(fake_bgr)
        assert gray.ndim == 2
        assert gray.dtype == np.uint8

    def test_to_grayscale_idempotent(self, fake_gray):
        out = to_grayscale(fake_gray)
        assert out.shape == fake_gray.shape

    def test_to_rgb(self, fake_bgr):
        rgb = to_rgb(fake_bgr)
        assert rgb.shape == fake_bgr.shape
        # BGR[0] == RGB[2]
        assert np.array_equal(fake_bgr[0, 0, 0], rgb[0, 0, 2])

    def test_to_bgr(self, fake_bgr):
        rgb = to_rgb(fake_bgr)
        bgr = to_bgr(rgb)
        assert np.array_equal(fake_bgr, bgr)

    def test_to_hsv(self, fake_bgr):
        hsv = to_hsv(fake_bgr)
        assert hsv.shape == fake_bgr.shape

    def test_to_lab(self, fake_bgr):
        lab = to_lab(fake_bgr)
        assert lab.shape == fake_bgr.shape

    def test_convert_color_identity(self, fake_bgr):
        same = convert_color(fake_bgr, "BGR", "BGR")
        assert np.array_equal(fake_bgr, same)

    def test_convert_color_invalid(self, fake_bgr):
        with pytest.raises(ValueError, match="不支持的色彩空间转换"):
            convert_color(fake_bgr, "GRAY", "HSV")

    def test_to_hsv_raises_on_gray(self, fake_gray):
        with pytest.raises(ValueError):
            to_hsv(fake_gray)

    def test_convert_invalid_ndim(self):
        bad = np.random.randint(0, 256, (10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="ndim"):
            convert_color(bad, "BGR", "RGB")


# ============ enhance ============

class TestEnhance:
    def test_brightness_increase(self, fake_bgr):
        bright = adjust_brightness(fake_bgr, alpha=1.0, beta=50)
        assert bright.dtype == np.uint8
        assert bright.shape == fake_bgr.shape
        # 亮度应整体增加
        assert bright.mean() > fake_bgr.mean()

    def test_brightness_decrease(self, fake_bgr):
        dark = adjust_brightness(fake_bgr, alpha=0.5, beta=0)
        assert dark.mean() < fake_bgr.mean()

    def test_alpha_negative(self, fake_bgr):
        with pytest.raises(ValueError, match="alpha"):
            adjust_brightness(fake_bgr, alpha=-1)

    def test_empty_image(self):
        with pytest.raises(ValueError, match="为空"):
            adjust_brightness(np.array([], dtype=np.uint8).reshape(0, 0, 3))


# ============ noise ============

class TestNoise:
    def test_salt_pepper_shape(self, fake_bgr):
        noisy = salt_pepper_noise(fake_bgr, 0.1, 0.1)
        assert noisy.shape == fake_bgr.shape
        assert noisy.dtype == np.uint8

    def test_salt_pepper_gray(self, fake_gray):
        noisy = salt_pepper_noise(fake_gray, 0.1, 0.05)
        assert noisy.shape == fake_gray.shape

    def test_salt_pepper_zero_prob(self, fake_bgr):
        noisy = salt_pepper_noise(fake_bgr, 0, 0)
        assert np.array_equal(noisy, fake_bgr)

    def test_salt_pepper_invalid_prob(self, fake_bgr):
        with pytest.raises(ValueError):
            salt_pepper_noise(fake_bgr, 1.5, 0.1)
        with pytest.raises(ValueError):
            salt_pepper_noise(fake_bgr, 0.1, -0.1)

    def test_gaussian_noise_shape(self, fake_bgr):
        noisy = gaussian_noise(fake_bgr, mean=0, sigma=25)
        assert noisy.shape == fake_bgr.shape
        assert noisy.dtype == np.uint8

    def test_gaussian_noise_gray(self, fake_gray):
        noisy = gaussian_noise(fake_gray, sigma=10)
        assert noisy.shape == fake_gray.shape

    def test_gaussian_noise_zero_sigma(self, fake_bgr):
        """sigma=0 应返回原图."""
        noisy = gaussian_noise(fake_bgr, sigma=0)
        assert np.array_equal(noisy, fake_bgr)

    def test_gaussian_noise_negative_sigma(self, fake_bgr):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_noise(fake_bgr, sigma=-1)

    def test_gaussian_noise_empty(self):
        with pytest.raises(ValueError, match="为空"):
            gaussian_noise(np.array([], dtype=np.uint8).reshape(0, 0, 3))

    def test_gaussian_noise_actually_adds_noise(self, fake_bgr):
        """sigma > 0 时输出应与原图不同."""
        noisy = gaussian_noise(fake_bgr, sigma=30)
        assert not np.array_equal(noisy, fake_bgr)


# ============ blur ============

class TestBlur:
    def test_motion_blur_shape(self, fake_bgr):
        blurred = motion_blur(fake_bgr, kernel_size=15, angle=45)
        assert blurred.shape == fake_bgr.shape
        assert blurred.dtype == np.uint8

    def test_motion_blur_gray(self, fake_gray):
        blurred = motion_blur(fake_gray, kernel_size=15)
        assert blurred.shape == fake_gray.shape

    def test_motion_blur_kernel_size_odd(self, fake_bgr):
        """偶数 kernel_size 应报错."""
        with pytest.raises(ValueError, match="奇数"):
            motion_blur(fake_bgr, kernel_size=4)

    def test_motion_blur_kernel_too_small(self, fake_bgr):
        with pytest.raises(ValueError, match=">=3"):
            motion_blur(fake_bgr, kernel_size=1)

    def test_motion_blur_actually_smooths(self, fake_bgr):
        """模糊后图像梯度应变小."""
        blurred = motion_blur(fake_bgr, kernel_size=21, angle=0)
        orig_std = fake_bgr.std()
        blur_std = blurred.std()
        assert blur_std < orig_std

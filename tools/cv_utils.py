"""
Re-Value Agent - 图像处理基建
包含 rembg 调用、Alpha Mask 提取、Drop Shadow 阴影生成算法
"""

import io
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# =============================================================================
# 抠图工具 (Matting)
# =============================================================================

class MattingTool:
    """
    抠图工具，支持多种后端:
    - rembg (u2net, u2netp, u2net_human_seg, silueta)
    - RMBG-1.4 (BRIA) - 高精度备选
    """

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._rembg_session = None

    def _get_rembg_session(self):
        """延迟初始化 rembg session"""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                self._rembg_session = new_session(model_name=self.model_name)
                logger.info("rembg session initialized with model: %s", self.model_name)
            except Exception as e:
                logger.error("Failed to initialize rembg: %s", e)
                raise
        return self._rembg_session

    def remove_background(
        self,
        image: Image.Image,
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
    ) -> Image.Image:
        """
        移除图像背景，返回 RGBA 图像

        Args:
            image: 输入 RGB 图像 (PIL Image)
            alpha_matting: 是否使用 alpha matting (边缘更精细)
            alpha_matting_foreground_threshold: 前景阈值
            alpha_matting_background_threshold: 背景阈值

        Returns:
            输出 RGBA 图像

        Raises:
            ValueError: 如果 Alpha 通道有效面积占比极低
        """
        try:
            from rembg import remove

            result = remove(
                image,
                session=self._get_rembg_session(),
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
            )

            # 验证 Alpha 通道
            if not self._validate_alpha(result):
                raise ValueError(
                    f"Alpha channel valid area too low: "
                    f"expected >5%, got {self._get_alpha_coverage(result):.2f}%"
                )

            return result

        except ImportError:
            logger.warning("rembg not installed, using fallback matting")
            return self._fallback_matting(image)

    def _validate_alpha(self, rgba_image: Image.Image) -> bool:
        """验证 Alpha 通道有效面积"""
        coverage = self._get_alpha_coverage(rgba_image)
        return coverage > 5.0  # 阈值 5%

    def _get_alpha_coverage(self, rgba_image: Image.Image) -> float:
        """计算 Alpha 通道有效面积占比"""
        if rgba_image.mode != "RGBA":
            return 0.0

        alpha = np.array(rgba_image.split()[-1])
        total_pixels = alpha.size
        valid_pixels = np.count_nonzero(alpha > 10)  # 忽略近乎透明的像素
        return (valid_pixels / total_pixels) * 100

    def _fallback_matting(self, image: Image.Image) -> Image.Image:
        """
        Fallback 抠图方案：简单的阈值分割
        当 rembg 不可用时使用
        """
        # 转换为 numpy 数组
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 使用 Otsu's 阈值分割
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 创建 alpha 通道
        alpha = thresh

        # 转为 RGBA
        rgba_array = np.dstack([img_array, alpha])
        return Image.fromarray(rgba_array.astype(np.uint8), "RGBA")


# =============================================================================
# 阴影渲染工具 (Drop Shadow)
# =============================================================================

class ShadowRenderer:
    """
    阴影渲染器，生成自然的投影效果
    """

    def __init__(
        self,
        blur_radius: int = 15,
        shadow_alpha: int = 100,
        offset_y: int = 10,
    ):
        self.blur_radius = blur_radius
        self.shadow_alpha = shadow_alpha
        self.offset_y = offset_y

    def generate_shadow(
        self,
        rgba_image: Image.Image,
        shadow_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """
        为 RGBA 图像生成阴影层

        Args:
            rgba_image: 输入 RGBA 图像
            shadow_color: 阴影颜色 (默认黑色)

        Returns:
            与原图等大的阴影层 (L 模式，灰度)
        """
        if rgba_image.mode != "RGBA":
            raise ValueError("Input image must be RGBA")

        # 提取 Alpha 通道作为 mask
        alpha = np.array(rgba_image.split()[-1])

        # 创建纯黑图像
        shadow = np.zeros_like(alpha)

        # 应用 Alpha mask
        shadow[alpha > 10] = self.shadow_alpha

        # 高斯模糊
        shadow_pil = Image.fromarray(shadow, mode="L")
        shadow_pil = shadow_pil.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # 向下偏移
        shadow_array = np.array(shadow_pil)
        shadow_array = np.roll(shadow_array, self.offset_y, axis=0)

        # 裁剪回原始尺寸
        shadow_array = shadow_array[:rgba_image.height, :rgba_image.width]

        return Image.fromarray(shadow_array, mode="L")

    def apply_shadow(
        self,
        rgba_image: Image.Image,
        background: Image.Image,
    ) -> Image.Image:
        """
        将阴影应用到背景图

        Args:
            rgba_image: 带有 Alpha 通道的商品图 (RGBA)
            background: 背景图 (RGB)

        Returns:
            合成后的图像 (RGB)
        """
        # 生成阴影层
        shadow = self.generate_shadow(rgba_image)

        # 确保背景与阴影尺寸匹配
        if background.size != rgba_image.size:
            background = background.resize(rgba_image.size, Image.LANCZOS)

        # 创建 RGB 版本的主体
        rgb_image = rgba_image.convert("RGB")

        # 将阴影作为底层放置
        background.paste(shadow, (0, 0), shadow)
        # 将主体覆盖在阴影上
        background.paste(rgb_image, (0, 0), rgba_image)

        return background


# =============================================================================
# 图像融合工具 (Fusion)
# =============================================================================

class ImageFusion:
    """
    图像融合工具，将抠图主体与新背景合成
    """

    def __init__(
        self,
        subject_height_ratio: float = 0.65,
        bottom_margin_ratio: float = 0.15,
    ):
        """
        Args:
            subject_height_ratio: 主体占背景高度的比例 (0.0-1.0)
            bottom_margin_ratio: 距离底部的边距比例 (0.0-1.0)
        """
        self.subject_height_ratio = subject_height_ratio
        self.bottom_margin_ratio = bottom_margin_ratio

    def compute_placement(
        self,
        subject_size: Tuple[int, int],
        background_size: Tuple[int, int],
        reference_box: Optional[list] = None,
    ) -> Tuple[int, int, int, int]:
        """
        计算主体在背景中的放置位置

        Args:
            subject_size: (width, height) 主体尺寸
            background_size: (width, height) 背景尺寸
            reference_box: 可选的 [x_min, y_min, x_max, y_max] 参考框

        Returns:
            (x, y, scaled_w, scaled_h) 放置位置和缩放后尺寸
        """
        bg_w, bg_h = background_size
        subj_w, subj_h = subject_size

        # 规则计算：按高度比例缩放
        target_height = int(bg_h * self.subject_height_ratio)
        scale = target_height / subj_h
        scaled_w = int(subj_w * scale)
        scaled_h = target_height

        # 水平居中
        x = (bg_w - scaled_w) // 2

        # 垂直偏下，距离底部留边距
        y = int(bg_h * (1 - self.bottom_margin_ratio)) - scaled_h

        # 校验参考框 (如果提供)
        if reference_box and self._validate_reference_box(reference_box, bg_w, bg_h):
            # 使用参考框计算的位置
            x_min, y_min, x_max, y_max = reference_box
            ref_width = int((x_max - x_min) * bg_w)
            ref_height = int((y_max - y_min) * bg_h)

            # 按参考框的比例缩放主体
            scale = min(ref_width / subj_w, ref_height / subj_h)
            scaled_w = int(subj_w * scale)
            scaled_h = int(subj_h * scale)

            # 使用参考框中心作为放置参考
            center_x = (x_min + x_max) / 2 * bg_w
            center_y = (y_min + y_max) / 2 * bg_h
            x = int(center_x - scaled_w / 2)
            y = int(center_y - scaled_h / 2)

        # 确保不超出边界
        x = max(0, min(x, bg_w - scaled_w))
        y = max(0, min(y, bg_h - scaled_h))

        return x, y, scaled_w, scaled_h

    def _validate_reference_box(
        self,
        reference_box: list,
        img_width: int,
        img_height: int,
    ) -> bool:
        """验证参考框是否合理"""
        if not reference_box or len(reference_box) != 4:
            return False

        x_min, y_min, x_max, y_max = reference_box

        # 检查是否在 0-1 范围内
        if not all(0 <= v <= 1 for v in [x_min, y_min, x_max, y_max]):
            return False

        # 检查宽高占比是否合理 (20% - 80%)
        width_ratio = x_max - x_min
        height_ratio = y_max - y_min
        if width_ratio < 0.2 or width_ratio > 0.8:
            return False
        if height_ratio < 0.2 or height_ratio > 0.8:
            return False

        # 检查是否越界
        if x_max <= x_min or y_max <= y_min:
            return False

        return True

    def fuse(
        self,
        subject_rgba: Image.Image,
        background: Image.Image,
        reference_box: Optional[list] = None,
        shadow_renderer: Optional[ShadowRenderer] = None,
    ) -> Image.Image:
        """
        将抠图主体融合到新背景

        Args:
            subject_rgba: 抠图后的 RGBA 图像
            background: 背景图 (RGB)
            reference_box: 可选的参考框
            shadow_renderer: 阴影渲染器 (可选)

        Returns:
            合成后的图像
        """
        # 缩放主体
        x, y, new_w, new_h = self.compute_placement(
            subject_rgba.size,
            background.size,
            reference_box,
        )

        # 缩放主体
        resized_subject = subject_rgba.resize((new_w, new_h), Image.LANCZOS)

        # 创建输出图像
        output = background.copy()

        # 生成阴影 (如果提供了 renderer)
        if shadow_renderer:
            shadow = shadow_renderer.generate_shadow(resized_subject)
            # 在放置主体下方添加阴影
            shadow_layer = Image.new("L", background.size, 0)
            shadow_layer.paste(shadow, (x, y))
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_renderer.blur_radius))

            # 将阴影混合到背景
            # 确保是 RGB 模式
            if output.mode != "RGB":
                output = output.convert("RGB")

            shadow_arr = np.array(shadow_layer)
            output_arr = np.array(output)

            # 叠加阴影
            shadow_arr = shadow_arr[:, :, np.newaxis] / 255.0  # (H, W, 1)
            shadow_color = np.array([30, 30, 30]) / 255.0  # (3,)
            # 扩展 shadow_color 到 (1, 1, 3) 以便广播
            output_arr = output_arr * (1 - shadow_arr * 0.3) + shadow_color[np.newaxis, np.newaxis, :] * shadow_arr * 255

            output = Image.fromarray(output_arr.astype(np.uint8))

        # 放置主体
        output.paste(resized_subject, (x, y), resized_subject)

        return output


# =============================================================================
# 工具函数
# =============================================================================

def validate_image(image: Image.Image, max_size: int = 4096) -> bool:
    """
    验证图像是否有效

    Args:
        image: PIL Image
        max_size: 最大尺寸限制

    Returns:
        是否有效
    """
    if image is None:
        return False

    w, h = image.size
    if w < 10 or h < 10:
        return False

    if w > max_size or h > max_size:
        logger.warning("Image size %dx%d exceeds max_size %d", w, h, max_size)
        return False

    return True


def resize_if_needed(
    image: Image.Image,
    max_size: int = 1024,
) -> Image.Image:
    """
    如果图像太大，按比例缩小

    Args:
        image: 输入图像
        max_size: 最大边长

    Returns:
        调整后的图像
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image

    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    return image.resize((new_w, new_h), Image.LANCZOS)

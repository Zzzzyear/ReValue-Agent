"""
Re-Value Agent - 文字渲染器
支持中文文字渲染到图像上，带阴影、描边、半透明背景
"""

import math
from PIL import Image, ImageDraw, ImageFont


class TextRenderer:
    """
    文字渲染器

    功能:
    - 中文字体支持
    - 半透明背景
    - 文字阴影
    - 自动换行
    - 位置控制
    """

    def __init__(
        self,
        font_path: str = None,
        default_font_size: int = 60,
    ):
        """
        Args:
            font_path: 字体文件路径，None则使用默认字体
            default_font_size: 默认字体大小
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
        self._font_cache = {}  # 缓存不同大小的字体

    def _get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """获取指定大小的字体，缓存以提高性能"""
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                # 尝试使用系统中文字体
                font = self._get_system_cjk_font(font_size)
        except Exception:
            # 回退到默认字体
            font = ImageFont.load_default()

        self._font_cache[font_size] = font
        return font

    def _get_system_cjk_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """尝试获取系统中文字体"""
        import os

        # Windows 常见中文字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑体
            "C:/Windows/Fonts/simsun.ttc",     # 宋体
            "C:/Windows/Fonts/arial.ttf",      # 备用
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        # Linux 常见路径
        linux_font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/System/Library/Fonts/PingFang.ttc",
        ]

        for path in linux_font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        # 最后尝试默认
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """
        自动换行

        Args:
            text: 原始文本
            font: 字体对象
            max_width: 最大宽度

        Returns:
            换行后的行列表
        """
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def render(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom_center",
        max_width_ratio: float = 0.85,
        font_size: int = None,
        background_color: tuple = (0, 0, 0, 180),
        text_color: tuple = (255, 255, 255),
        shadow_color: tuple = (0, 0, 0),
        padding: int = 20,
        line_spacing: float = 1.2,
    ) -> Image.Image:
        """
        将文字渲染到图像上

        Args:
            image: 背景图像 (PIL Image)
            text: 要渲染的文字
            position: 位置 (bottom_center | top_center | center | overlay)
            max_width_ratio: 文字最大宽度占图像宽度的比例
            font_size: 字体大小，None则自动计算
            background_color: 背景色 (RGBA)
            text_color: 文字颜色 (RGB)
            shadow_color: 阴影颜色
            padding: 内边距
            line_spacing: 行间距倍数

        Returns:
            渲染后的图像
        """
        if font_size is None:
            font_size = self.default_font_size

        font = self._get_font(font_size)
        img_w, img_h = image.size

        # 计算最大宽度
        max_width = int(img_w * max_width_ratio)

        # 换行处理
        lines = self._wrap_text(text, font, max_width - 2 * padding)

        # 计算文字尺寸
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = font.getbbox(line)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            line_heights.append(lh)
            line_widths.append(lw)

        total_height = sum(line_heights) * line_spacing
        max_line_width = max(line_widths) if line_widths else 0

        # 计算背景尺寸
        bg_width = max_line_width + 2 * padding
        bg_height = total_height + 2 * padding

        # 计算位置
        if position == "bottom_center":
            x = (img_w - bg_width) // 2
            y = img_h - bg_height - int(img_h * 0.05)  # 距底部5%
        elif position == "top_center":
            x = (img_w - bg_width) // 2
            y = int(img_h * 0.05)
        elif position == "center":
            x = (img_w - bg_width) // 2
            y = (img_h - bg_height) // 2
        elif position == "bottom_left":
            x = int(img_w * 0.05)
            y = img_h - bg_height - int(img_h * 0.05)
        elif position == "bottom_right":
            x = img_w - bg_width - int(img_w * 0.05)
            y = img_h - bg_height - int(img_h * 0.05)
        else:
            x = (img_w - bg_width) // 2
            y = img_h - bg_height - int(img_h * 0.05)

        # 确保在边界内
        x = max(0, min(x, img_w - bg_width))
        y = max(0, min(y, img_h - bg_height))

        # 创建输出图像
        if image.mode != "RGBA":
            output = image.convert("RGBA")
        else:
            output = image.copy()

        # 创建文字层
        text_layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # 绘制背景
        draw.rounded_rectangle(
            [x, y, x + bg_width, y + bg_height],
            radius=int(min(bg_width, bg_height) * 0.15),
            fill=background_color,
        )

        # 绘制文字
        current_y = y + padding
        for i, line in enumerate(lines):
            bbox = font.getbbox(line)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]

            # 居中计算
            line_x = x + (bg_width - lw) // 2

            # 绘制阴影
            shadow_offset = 2
            draw.text(
                (line_x + shadow_offset, current_y + shadow_offset),
                line,
                font=font,
                fill=(*shadow_color, 180),
            )

            # 绘制主文字
            draw.text(
                (line_x, current_y),
                line,
                font=font,
                fill=text_color,
            )

            current_y += lh * line_spacing

        # 合成
        output = Image.alpha_composite(output, text_layer)

        return output.convert("RGB")


def render_title_on_image(
    image: Image.Image,
    title: str,
    position: str = "bottom_center",
    max_chars: int = 25,
) -> Image.Image:
    """
    便捷函数：将标题渲染到图像上

    Args:
        image: 背景图像
        title: 标题文本
        position: 位置
        max_chars: 标题最大字符数，超出截断

    Returns:
        渲染后的图像
    """
    # 凝练标题：截断过长部分，保留核心信息
    if len(title) > max_chars:
        # 尝试在标点处截断
        for p in ['｜', '!', '！', '—', '-', ' ', '✨', '💜']:
            if p in title and title.index(p) < max_chars - 3:
                short_title = title[:title.index(p)]
                if len(short_title) < max_chars - 3:
                    title = short_title + '...'
                    break
        else:
            title = title[:max_chars - 3] + '...'

    renderer = TextRenderer()
    return renderer.render(
        image=image,
        text=title,
        position=position,
        max_width_ratio=0.85,
        font_size=48,
        background_color=(0, 0, 0, 160),
        text_color=(255, 255, 255),
    )

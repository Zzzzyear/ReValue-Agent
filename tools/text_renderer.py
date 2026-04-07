"""
Re-Value Agent - 文字渲染器
支持中文文字 + Emoji 渲染到图像上，带阴影、描边、半透明背景
"""

import logging
import math
import re
from PIL import Image, ImageDraw, ImageFont

try:
    from pilmoji import Pilmoji

    _HAS_PILMOJI = True
except ImportError:
    _HAS_PILMOJI = False

logger = logging.getLogger(__name__)

_pilmoji_checked = False


def _check_pilmoji_cdn(timeout: float = 3.0) -> bool:
    """首次调用时检测 emojicdn.elk.sh 是否可达，不可达则全局禁用 pilmoji"""
    global _HAS_PILMOJI, _pilmoji_checked
    if _pilmoji_checked:
        return _HAS_PILMOJI
    _pilmoji_checked = True

    if not _HAS_PILMOJI:
        return False

    import urllib.request
    try:
        urllib.request.urlopen("https://emojicdn.elk.sh/%E2%9C%A8?style=twitter", timeout=timeout)
        logger.info("pilmoji CDN reachable, emoji rendering enabled")
        return True
    except Exception:
        logger.warning("pilmoji CDN unreachable, falling back to plain text rendering (emoji will be stripped)")
        _HAS_PILMOJI = False
        return False

# 匹配 Emoji 及常见符号字符的正则（仅在 pilmoji 不可用时作为 fallback）
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "\U0000231A-\U0000231B"
    "\U000023E9-\U000023F3"
    "\U000023F8-\U000023FA"
    "\U0000200B"
    "\U0000200D"
    "\U0000FE00-\U0000FE0F"
    "\U00003030"
    "\U0000303D"
    "]+",
    flags=re.UNICODE,
)


class TextRenderer:
    """
    文字渲染器

    功能:
    - 中文字体 + Emoji 混合渲染（通过 pilmoji）
    - 半透明背景
    - 文字阴影
    - 自动换行
    - 位置控制
    - 无 pilmoji 时自动降级为过滤 Emoji
    """

    def __init__(
        self,
        font_path: str = None,
        default_font_size: int = 60,
    ):
        self.font_path = font_path
        self.default_font_size = default_font_size
        self._font_cache = {}

    @staticmethod
    def _strip_emoji(text: str) -> str:
        """移除 Emoji 及不可渲染的特殊符号（fallback 用）"""
        cleaned = _EMOJI_RE.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = self._get_system_cjk_font(font_size)
        except Exception:
            font = ImageFont.load_default()

        self._font_cache[font_size] = font
        return font

    def _get_system_cjk_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        import os

        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        linux_font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]

        for path in linux_font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        return ImageFont.load_default()

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont) -> tuple:
        """测量文本宽高，支持 Emoji（网络不可用时自动降级）"""
        if _HAS_PILMOJI:
            try:
                tmp = Image.new("RGBA", (1, 1))
                with Pilmoji(tmp) as pmj:
                    return pmj.getsize(text, font=font)
            except Exception:
                pass
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """自动换行（支持 Emoji 宽度计算）"""
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            w, _ = self._measure_text(test_line, font)

            if w <= max_width:
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
        将文字渲染到图像上（自动处理 Emoji）

        Args:
            image: 背景图像 (PIL Image)
            text: 要渲染的文字（可含 Emoji）
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
        _check_pilmoji_cdn()
        use_pilmoji = _HAS_PILMOJI
        if not use_pilmoji:
            text = self._strip_emoji(text)
        if not text or not text.strip():
            return image

        if font_size is None:
            font_size = self.default_font_size

        font = self._get_font(font_size)
        ascent, descent = font.getmetrics()
        fixed_line_height = ascent + descent

        img_w, img_h = image.size
        max_width = int(img_w * max_width_ratio)

        lines = self._wrap_text(text, font, max_width - 2 * padding)

        line_widths = [self._measure_text(line, font)[0] for line in lines]
        max_line_width = max(line_widths) if line_widths else 0

        n = len(lines)
        total_height = fixed_line_height * ((n - 1) * line_spacing + 1)

        bg_width = max_line_width + 2 * padding
        bg_height = int(total_height + 2 * padding)

        if position == "bottom_center":
            x = (img_w - bg_width) // 2
            y = img_h - bg_height - int(img_h * 0.05)
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

        x = max(0, min(x, img_w - bg_width))
        y = max(0, min(y, img_h - bg_height))

        if image.mode != "RGBA":
            output = image.convert("RGBA")
        else:
            output = image.copy()

        text_layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        draw.rounded_rectangle(
            [x, y, x + bg_width, y + bg_height],
            radius=int(min(bg_width, bg_height) * 0.15),
            fill=background_color,
        )

        current_y = y + padding
        shadow_offset = 2

        line_advance = int(fixed_line_height * line_spacing)

        if use_pilmoji:
            try:
                with Pilmoji(text_layer) as pmj:
                    for i, line in enumerate(lines):
                        line_x = x + (bg_width - line_widths[i]) // 2
                        pmj.text(
                            (line_x + shadow_offset, current_y + shadow_offset),
                            line, font=font, fill=(*shadow_color, 180),
                        )
                        pmj.text(
                            (line_x, current_y),
                            line, font=font, fill=text_color,
                        )
                        current_y += line_advance
            except Exception as e:
                logger.warning("pilmoji rendering failed, falling back to plain text: %s", e)
                use_pilmoji = False
                text_layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(text_layer)
                draw.rounded_rectangle(
                    [x, y, x + bg_width, y + bg_height],
                    radius=int(min(bg_width, bg_height) * 0.15),
                    fill=background_color,
                )

        if not use_pilmoji:
            stripped_lines = [self._strip_emoji(l) for l in lines]
            stripped_widths = [font.getbbox(l)[2] - font.getbbox(l)[0] if l else 0 for l in stripped_lines]
            current_y = y + padding
            for i, line in enumerate(stripped_lines):
                if not line:
                    current_y += line_advance
                    continue
                line_x = x + (bg_width - stripped_widths[i]) // 2
                draw.text(
                    (line_x + shadow_offset, current_y + shadow_offset),
                    line, font=font, fill=(*shadow_color, 180),
                )
                draw.text(
                    (line_x, current_y),
                    line, font=font, fill=text_color,
                )
                current_y += line_advance

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
        title: 标题文本（可含 Emoji）
        position: 位置
        max_chars: 标题最大字符数，超出截断

    Returns:
        渲染后的图像
    """
    if not title or not title.strip():
        return image

    if len(title) > max_chars:
        for p in ['｜', '!', '！', '—', '-', ' ']:
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

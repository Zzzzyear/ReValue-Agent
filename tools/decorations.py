"""
Re-Value Agent - 装饰元素绘制器
支持两种模式：
1. Emoji 贴纸模式（推荐）：大模型输出 emoji 字符，用本地 Noto Color Emoji 字体渲染
2. 旧 type 模式（向后兼容）：使用 PIL 手绘 sparkle/star/heart 等图形
"""

import logging
import math
import os
import random
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_EMOJI_PIXEL_SIZE_MAP = {"small": 80, "medium": 120, "large": 180}

_EMOJI_FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-cjk/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
]


def _find_emoji_font() -> Optional[str]:
    for p in _EMOJI_FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


def _render_emoji_to_image(emoji_char: str, target_px: int) -> Optional[Image.Image]:
    """
    将单个 emoji 字符渲染为 RGBA 透明底 Image，缩放到 target_px 大小。
    使用本地 Noto Color Emoji 字体 + Pillow embedded_color 渲染，无需联网。
    """
    font_path = _find_emoji_font()
    if font_path is None:
        logger.warning("No color emoji font found, emoji rendering disabled")
        return None

    # Noto Color Emoji 内置 128px 位图，用 109 可获得最清晰渲染
    render_size = 109
    padding = render_size
    canvas = render_size + padding * 2
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))

    try:
        font = ImageFont.truetype(font_path, render_size)
    except Exception as e:
        logger.warning("Failed to load emoji font: %s", e)
        return None

    draw = ImageDraw.Draw(img)
    try:
        draw.text((padding, padding), emoji_char, font=font,
                  fill=(255, 255, 255, 255), embedded_color=True)
    except Exception as e:
        logger.warning("Emoji draw failed for '%s': %s", emoji_char, e)
        return None

    bbox = img.getbbox()
    if bbox is None:
        return None

    cropped = img.crop(bbox)
    cw, ch = cropped.size
    if cw == 0 or ch == 0:
        return None

    aspect = cw / ch
    if aspect >= 1:
        new_w = target_px
        new_h = max(1, int(target_px / aspect))
    else:
        new_h = target_px
        new_w = max(1, int(target_px * aspect))

    return cropped.resize((new_w, new_h), Image.LANCZOS)


class DecorationRenderer:
    """
    装饰元素渲染器

    新模式（emoji 贴纸）：
        creative_elements 中每个元素包含 "emoji" 字段，如 {"emoji": "✨", ...}

    旧模式（向后兼容）：
        creative_elements 中每个元素包含 "type" 字段，如 {"type": "sparkle", ...}
    """

    _FONT_PATHS = [
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    POSITION_MAP = {
        "top-left": lambda w, h, ew, eh: (int(w * 0.05), int(h * 0.05)),
        "top-right": lambda w, h, ew, eh: (int(w * 0.85), int(h * 0.05)),
        "bottom-left": lambda w, h, ew, eh: (int(w * 0.05), int(h * 0.8)),
        "bottom-right": lambda w, h, ew, eh: (int(w * 0.85), int(h * 0.8)),
        "top-center": lambda w, h, ew, eh: ((w - ew) // 2, int(h * 0.03)),
        "bottom-center": lambda w, h, ew, eh: ((w - ew) // 2, int(h * 0.85)),
        "center": lambda w, h, ew, eh: ((w - ew) // 2, (h - eh) // 2),
    }

    def __init__(self, **kwargs):
        self._emoji_cache: Dict[str, Image.Image] = {}

    # ── Emoji 渲染 ────────────────────────────────────────

    def _get_emoji_image(self, emoji_char: str, size: str) -> Optional[Image.Image]:
        """获取 emoji 的 RGBA 贴图，带缓存"""
        target_px = _EMOJI_PIXEL_SIZE_MAP.get(size, 120)
        cache_key = f"{emoji_char}_{target_px}"
        if cache_key in self._emoji_cache:
            return self._emoji_cache[cache_key]

        img = _render_emoji_to_image(emoji_char, target_px)
        if img is not None:
            self._emoji_cache[cache_key] = img
        return img

    def _paste_emoji(
        self,
        image: Image.Image,
        emoji_img: Image.Image,
        position: str,
        count: int,
    ) -> Image.Image:
        """将 emoji 贴图粘贴到图像的指定位置"""
        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        ew, eh = emoji_img.size
        w, h = img.size
        base_x, base_y = self._get_position(position, (w, h), (ew, eh))

        rng = random.Random(42)

        for i in range(count):
            if count > 1:
                ox = rng.randint(-w // 12, w // 12)
                oy = rng.randint(-h // 12, h // 12)
            else:
                ox, oy = 0, 0

            px = max(0, min(base_x + ox, w - ew))
            py = max(0, min(base_y + oy, h - eh))

            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            layer.paste(emoji_img, (px, py), emoji_img)
            img = Image.alpha_composite(img, layer)

        return img

    # ── 工具方法 ──────────────────────────────────────────

    @classmethod
    def _get_font(cls, size: int) -> ImageFont.FreeTypeFont:
        for path in cls._FONT_PATHS:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _get_position(position: str, img_size: tuple, elem_size: tuple) -> tuple:
        w, h = img_size
        ew, eh = elem_size
        if position in DecorationRenderer.POSITION_MAP:
            return DecorationRenderer.POSITION_MAP[position](w, h, ew, eh)
        return DecorationRenderer.POSITION_MAP["top-right"](w, h, ew, eh)

    @staticmethod
    def _get_size(size: str, base: int) -> int:
        size_map = {"small": 0.7, "medium": 1.0, "large": 1.4}
        return int(base * size_map.get(size, 1.0))

    # ── 主入口 ────────────────────────────────────────────

    def render_decoration(
        self,
        image: Image.Image,
        decoration_type: str = "",
        position: str = "top-right",
        count: int = 1,
        size: str = "medium",
        color: tuple = None,
        text: str = None,
        style: str = "rounded_rect",
        emoji: str = None,
    ) -> Image.Image:
        """渲染单个装饰元素（优先 emoji，回退到旧 type 逻辑）"""
        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        # ── 优先走 emoji 路径 ──
        if emoji:
            emoji_img = self._get_emoji_image(emoji, size)
            if emoji_img is not None:
                img = self._paste_emoji(img, emoji_img, position, count)
                return img.convert("RGB")
            logger.warning("Emoji '%s' render failed, skipping", emoji)
            return img.convert("RGB")

        # ── 旧 type 兼容路径 ──
        if decoration_type == "sparkle":
            img = self._draw_sparkle(img, position, count, size, color)
        elif decoration_type == "star":
            img = self._draw_star(img, position, count, size, color)
        elif decoration_type == "heart":
            img = self._draw_heart(img, position, count, size, color)
        elif decoration_type == "badge":
            img = self._draw_badge(img, position, text, style, color)
        elif decoration_type == "price_tag":
            img = self._draw_price_tag(img, position, text, color)
        elif decoration_type == "ribbon":
            img = self._draw_ribbon(img, position, color)
        elif decoration_type == "circle_decoration":
            img = self._draw_circle(img, position, size, color)

        return img.convert("RGB")

    # ── 批量渲染 ──────────────────────────────────────────

    def render_elements(
        self,
        image: Image.Image,
        elements: list,
    ) -> Image.Image:
        """
        批量渲染装饰元素

        Args:
            image: 背景图像
            elements: 装饰元素列表，支持两种格式：
                emoji 模式: {"emoji": "✨", "position": "top-right", "count": 2, "size": "small"}
                旧 type 模式: {"type": "sparkle", "position": "top-right", "count": 3, "size": "small"}
        """
        result = image.copy()

        for elem in elements:
            result = self.render_decoration(
                image=result,
                decoration_type=elem.get("type", ""),
                position=elem.get("position", "top-right"),
                count=elem.get("count", 1),
                size=elem.get("size", "medium"),
                color=elem.get("color"),
                text=elem.get("text", ""),
                style=elem.get("style", "rounded_rect"),
                emoji=elem.get("emoji"),
            )

        return result

    # ── 旧 type 兼容：PIL 手绘方法 ────────────────────────

    def _draw_sparkle(self, image, position, count, size, color):
        if color is None:
            color = (255, 215, 0)
        base_size = 50
        actual_size = self._get_size(size, base_size)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        base_x, base_y = self._get_position(position, (w, h), (actual_size * 4, actual_size * 4))
        rng = random.Random(42)
        for _ in range(count):
            ox = rng.randint(-w // 10, w // 10)
            oy = rng.randint(-h // 10, h // 10)
            x, y = base_x + ox, base_y + oy
            arm = actual_size
            lw = max(3, actual_size // 5)
            draw.line([(x - arm, y), (x + arm, y)], fill=(*color, 250), width=lw)
            draw.line([(x, y - arm), (x, y + arm)], fill=(*color, 250), width=lw)
            sa = arm * 0.4
            draw.line([(x - sa, y - sa), (x + sa, y + sa)], fill=(*color, 200), width=max(2, lw // 2))
            draw.line([(x + sa, y - sa), (x - sa, y + sa)], fill=(*color, 200), width=max(2, lw // 2))
        return Image.alpha_composite(img, overlay)

    def _draw_star(self, image, position, count, size, color):
        if color is None:
            color = (255, 255, 200)
        actual_size = self._get_size(size, 60)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        for i in range(count):
            angle = (2 * math.pi * i) / max(count, 1)
            rx = w // 8 if count > 1 else 0
            ry = h // 8 if count > 1 else 0
            bx, by = self._get_position(position, (w, h), (actual_size * 2, actual_size * 2))
            x = bx + int(rx * math.cos(angle))
            y = by + int(ry * math.sin(angle))
            points = []
            for j in range(10):
                a = math.pi / 2 + (2 * math.pi * j) / 10
                r = actual_size if j % 2 == 0 else actual_size * 0.4
                points.append((x + int(r * math.cos(a)), y - int(r * math.sin(a))))
            draw.polygon(points, fill=(*color, 220))
        return Image.alpha_composite(img, overlay)

    def _draw_heart(self, image, position, count, size, color):
        if color is None:
            color = (255, 100, 150)
        actual_size = self._get_size(size, 55)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        rng = random.Random(123)
        for _ in range(count):
            ox = rng.randint(-w // 12, w // 12)
            oy = rng.randint(-h // 12, h // 12)
            bx, by = self._get_position(position, (w, h), (actual_size * 2, actual_size * 2))
            cx, cy = bx + ox, by + oy
            r = actual_size // 2
            draw.ellipse([cx - r - r // 2, cy - r // 2, cx - r // 2, cy + r], fill=(*color, 200))
            draw.ellipse([cx + r // 2, cy - r // 2, cx + r + r // 2, cy + r], fill=(*color, 200))
            draw.polygon([(cx - r - r // 2, cy), (cx + r + r // 2, cy), (cx, cy + r + r // 2)], fill=(*color, 200))
        return Image.alpha_composite(img, overlay)

    def _draw_badge(self, image, position, text, style, bg_color):
        if bg_color is None:
            bg_color = (255, 100, 100)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        bw, bh = min(w // 4, 200), min(h // 8, 60)
        x, y = self._get_position(position, (w, h), (bw, bh))
        if style == "rounded_rect":
            draw.rounded_rectangle([x, y, x + bw, y + bh], radius=bh // 3, fill=(*bg_color, 230))
        elif style == "circle":
            cx, cy = x + bw // 2, y + bh // 2
            r = min(bw, bh) // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*bg_color, 230))
        font = self._get_font(bh * 2 // 3)
        if text:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x + (bw - tw) // 2, y + (bh - th) // 2 - 2), text, font=font, fill=(255, 255, 255))
        return Image.alpha_composite(img, overlay)

    def _draw_price_tag(self, image, position, text, bg_color):
        if bg_color is None:
            bg_color = (255, 50, 50)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        tw_, th_ = min(w // 5, 180), min(h // 7, 50)
        x, y = self._get_position(position, (w, h), (tw_, th_))
        draw.rounded_rectangle([x, y, x + tw_, y + th_], radius=th_ // 4, fill=(*bg_color, 240))
        font = self._get_font(th_ * 3 // 5)
        if text:
            bbox = font.getbbox(text)
            txw, txh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x + (tw_ - txw) // 2
            ty = y + (th_ - txh) // 2 - 2
            draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 128))
            draw.text((tx, ty), text, font=font, fill=(255, 255, 255))
        return Image.alpha_composite(img, overlay)

    def _draw_ribbon(self, image, position, color):
        if color is None:
            color = (255, 100, 100)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        rw, rh = w // 3, h // 6
        x, y = self._get_position(position, (w, h), (rw, rh))
        draw.polygon([
            (x, y), (x + rw, y), (x + rw - rh // 3, y + rh // 2),
            (x + rw, y + rh), (x, y + rh), (x + rh // 3, y + rh // 2),
        ], fill=(*color, 200))
        return Image.alpha_composite(img, overlay)

    def _draw_circle(self, image, position, size, color):
        if color is None:
            color = (255, 215, 0, 128)
        actual_size = self._get_size(size, 50)
        img = image.copy() if image.mode == "RGBA" else image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        x, y = self._get_position(position, (w, h), (actual_size * 2, actual_size * 2))
        draw.ellipse([x, y, x + actual_size, y + actual_size], fill=(*color[:3], 100))
        inner = actual_size // 3
        draw.ellipse([
            x + actual_size // 2 - inner // 2, y + actual_size // 2 - inner // 2,
            x + actual_size // 2 + inner // 2, y + actual_size // 2 + inner // 2,
        ], fill=(*color[:3], 180))
        return Image.alpha_composite(img, overlay)

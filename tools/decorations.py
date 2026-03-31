"""
Re-Value Agent - 装饰元素绘制器
使用 PIL 绘制各类装饰元素：闪光、星星、心形、徽章、价格标签、丝带、圆形装饰
"""

import math
from PIL import Image, ImageDraw


class DecorationRenderer:
    """
    装饰元素渲染器

    支持的类型:
    - sparkle: 闪光/星星
    - star: 星形
    - heart: 心形
    - badge: 徽章
    - price_tag: 价格标签
    - ribbon: 丝带
    - circle_decoration: 圆形装饰
    """

    # 位置映射
    POSITION_MAP = {
        "top-left": lambda w, h, ew, eh: (int(w * 0.05), int(h * 0.05)),
        "top-right": lambda w, h, ew, eh: (int(w * 0.85), int(h * 0.05)),
        "bottom-left": lambda w, h, ew, eh: (int(w * 0.05), int(h * 0.8)),
        "bottom-right": lambda w, h, ew, eh: (int(w * 0.85), int(h * 0.8)),
        "top-center": lambda w, h, ew, eh: ((w - ew) // 2, int(h * 0.03)),
        "bottom-center": lambda w, h, ew, eh: ((w - ew) // 2, int(h * 0.85)),
        "center": lambda w, h, ew, eh: ((w - ew) // 2, (h - eh) // 2),
    }

    @staticmethod
    def _get_position(position: str, img_size: tuple, elem_size: tuple) -> tuple:
        """计算元素位置"""
        w, h = img_size
        ew, eh = elem_size
        if position in DecorationRenderer.POSITION_MAP:
            return DecorationRenderer.POSITION_MAP[position](w, h, ew, eh)
        # 默认右上
        return DecorationRenderer.POSITION_MAP["top-right"](w, h, ew, eh)

    @staticmethod
    def _get_size(size: str, base: int) -> int:
        """根据size字符串获取实际大小"""
        size_map = {"small": 0.7, "medium": 1.0, "large": 1.4}
        return int(base * size_map.get(size, 1.0))

    @classmethod
    def render_decoration(
        cls,
        image: Image.Image,
        decoration_type: str,
        position: str = "top-right",
        count: int = 1,
        size: str = "medium",
        color: tuple = None,
        text: str = None,
        style: str = "rounded_rect",
    ) -> Image.Image:
        """
        渲染单个装饰元素

        Args:
            image: 背景图像
            decoration_type: 类型 (sparkle/star/heart/badge/price_tag/ribbon/circle_decoration)
            position: 位置
            count: 数量 (sparkle/star/heart有效)
            size: 大小 (small/medium/large)
            color: 颜色 (RGB)，None则使用默认值
            text: 文字 (badge/price_tag有效)
            style: 样式 (badge有效)

        Returns:
            渲染后的图像
        """
        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        # 根据类型渲染
        if decoration_type == "sparkle":
            img = cls.draw_sparkle(img, position, count, size, color)
        elif decoration_type == "star":
            img = cls.draw_star(img, position, count, size, color)
        elif decoration_type == "heart":
            img = cls.draw_heart(img, position, count, size, color)
        elif decoration_type == "badge":
            img = cls.draw_badge(img, position, text, style, color)
        elif decoration_type == "price_tag":
            img = cls.draw_price_tag(img, position, text, color)
        elif decoration_type == "ribbon":
            img = cls.draw_ribbon(img, position, color)
        elif decoration_type == "circle_decoration":
            img = cls.draw_circle(img, position, size, color)

        return img.convert("RGB")

    @classmethod
    def draw_sparkle(
        cls,
        image: Image.Image,
        position: str = "top-right",
        count: int = 3,
        size: str = "small",
        color: tuple = None,
    ) -> Image.Image:
        """
        绘制闪光装饰

        Args:
            image: 背景图像
            position: 位置
            count: 数量
            size: 大小
            color: 颜色 (RGB)

        Returns:
            带闪光的图像
        """
        if color is None:
            color = (255, 215, 0)  # 金色

        # 增大基础尺寸，使装饰更显眼
        base_size = 50
        actual_size = cls._get_size(size, base_size)

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        # 计算基础位置
        base_x, base_y = cls._get_position(position, (w, h), (actual_size * 4, actual_size * 4))

        # 绘制多个闪光
        import random
        random.seed(42)  # 固定种子保证一致性

        for i in range(count):
            # 随机偏移
            offset_x = random.randint(-w // 10, w // 10)
            offset_y = random.randint(-h // 10, h // 10)
            x = base_x + offset_x
            y = base_y + offset_y

            # 绘制十字形闪光
            arm_length = actual_size
            line_width = max(3, actual_size // 5)

            # 主十字（增加透明度使更显眼）
            draw.line(
                [(x - arm_length, y), (x + arm_length, y)],
                fill=(*color, 250),
                width=line_width,
            )
            draw.line(
                [(x, y - arm_length), (x, y + arm_length)],
                fill=(*color, 250),
                width=line_width,
            )

            # 对角线小十字
            small_arm = arm_length * 0.4
            draw.line(
                [(x - small_arm, y - small_arm), (x + small_arm, y + small_arm)],
                fill=(*color, 200),
                width=max(2, line_width // 2),
            )
            draw.line(
                [(x + small_arm, y - small_arm), (x - small_arm, y + small_arm)],
                fill=(*color, 200),
                width=max(2, line_width // 2),
            )

        return Image.alpha_composite(img, overlay)

    @classmethod
    def draw_star(
        cls,
        image: Image.Image,
        position: str = "top-right",
        count: int = 1,
        size: str = "medium",
        color: tuple = None,
    ) -> Image.Image:
        """绘制星形装饰"""
        if color is None:
            color = (255, 255, 200)  # 淡黄色

        # 增大基础尺寸
        base_size = 60
        actual_size = cls._get_size(size, base_size)

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        for i in range(count):
            # 计算位置（分散开）
            angle = (2 * math.pi * i) / max(count, 1)
            radius_x = w // 8 if count > 1 else 0
            radius_y = h // 8 if count > 1 else 0
            offset_x = int(radius_x * math.cos(angle))
            offset_y = int(radius_y * math.sin(angle))

            base_x, base_y = cls._get_position(position, (w, h), (actual_size * 2, actual_size * 2))
            x = base_x + offset_x
            y = base_y + offset_y

            # 绘制五角星
            cls._draw_five_point_star(draw, x, y, actual_size, color)

        return Image.alpha_composite(img, overlay)

    @staticmethod
    def _draw_five_point_star(
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        radius: int,
        color: tuple,
    ):
        """绘制五角星"""
        points = []
        for i in range(10):
            angle = math.pi / 2 + (2 * math.pi * i) / 10
            r = radius if i % 2 == 0 else radius * 0.4
            x = cx + int(r * math.cos(angle))
            y = cy - int(r * math.sin(angle))
            points.append((x, y))
        draw.polygon(points, fill=(*color, 220))

    @classmethod
    def draw_heart(
        cls,
        image: Image.Image,
        position: str = "top-right",
        count: int = 1,
        size: str = "medium",
        color: tuple = None,
    ) -> Image.Image:
        """绘制心形装饰"""
        if color is None:
            color = (255, 100, 150)  # 粉色

        # 增大基础尺寸
        base_size = 55
        actual_size = cls._get_size(size, base_size)

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        import random
        random.seed(123)

        for i in range(count):
            offset_x = random.randint(-w // 12, w // 12)
            offset_y = random.randint(-h // 12, h // 12)
            base_x, base_y = cls._get_position(position, (w, h), (actual_size * 2, actual_size * 2))
            x = base_x + offset_x
            y = base_y + offset_y

            cls._draw_heart_shape(draw, x, y, actual_size, color)

        return Image.alpha_composite(img, overlay)

    @staticmethod
    def _draw_heart_shape(
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        size: int,
        color: tuple,
    ):
        """绘制心形"""
        # 心形由两个圆和一个三角形组成
        r = size // 2

        # 左半圆
        draw.ellipse(
            [cx - r - r // 2, cy - r // 2, cx - r // 2, cy + r],
            fill=(*color, 200),
        )
        # 右半圆
        draw.ellipse(
            [cx + r // 2, cy - r // 2, cx + r + r // 2, cy + r],
            fill=(*color, 200),
        )
        # 下部三角形
        draw.polygon(
            [
                (cx - r - r // 2, cy),
                (cx + r + r // 2, cy),
                (cx, cy + r + r // 2),
            ],
            fill=(*color, 200),
        )

    @classmethod
    def draw_badge(
        cls,
        image: Image.Image,
        position: str = "bottom-left",
        text: str = "HOT",
        style: str = "rounded_rect",
        bg_color: tuple = None,
    ) -> Image.Image:
        """绘制徽章"""
        if bg_color is None:
            bg_color = (255, 100, 100)  # 红色

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        # 计算徽章尺寸
        badge_w = min(w // 4, 200)
        badge_h = min(h // 8, 60)

        x, y = cls._get_position(position, (w, h), (badge_w, badge_h))

        if style == "rounded_rect":
            draw.rounded_rectangle(
                [x, y, x + badge_w, y + badge_h],
                radius=badge_h // 3,
                fill=(*bg_color, 230),
            )
        elif style == "circle":
            cx, cy = x + badge_w // 2, y + badge_h // 2
            r = min(badge_w, badge_h) // 2
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(*bg_color, 230),
            )
            badge_w = badge_h = r * 2
        elif style == "star":
            cls._draw_five_point_star(draw, x + badge_w // 2, y + badge_h // 2, badge_h // 2, bg_color)
            return Image.alpha_composite(img, overlay).convert("RGB")

        # 在徽章上写文字
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", badge_h * 2 // 3)
        except Exception:
            font = ImageFont.load_default()

        # 计算文字位置居中
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x + (badge_w - text_w) // 2
        text_y = y + (badge_h - text_h) // 2 - 2

        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=(255, 255, 255),
        )

        return Image.alpha_composite(img, overlay).convert("RGB")

    @classmethod
    def draw_price_tag(
        cls,
        image: Image.Image,
        position: str = "bottom-left",
        text: str = "2.8w",
        bg_color: tuple = None,
    ) -> Image.Image:
        """绘制价格标签"""
        if bg_color is None:
            bg_color = (255, 50, 50)  # 深红色

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        # 价格标签尺寸
        tag_w = min(w // 5, 180)
        tag_h = min(h // 7, 50)

        x, y = cls._get_position(position, (w, h), (tag_w, tag_h))

        # 绘制圆角矩形
        draw.rounded_rectangle(
            [x, y, x + tag_w, y + tag_h],
            radius=tag_h // 4,
            fill=(*bg_color, 240),
        )

        # 绘制文字
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", tag_h * 3 // 5)
        except Exception:
            font = ImageFont.load_default()

        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x + (tag_w - text_w) // 2
        text_y = y + (tag_h - text_h) // 2 - 2

        # 文字阴影
        draw.text(
            (text_x + 1, text_y + 1),
            text,
            font=font,
            fill=(0, 0, 0, 128),
        )
        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=(255, 255, 255),
        )

        return Image.alpha_composite(img, overlay).convert("RGB")

    @classmethod
    def draw_ribbon(
        cls,
        image: Image.Image,
        position: str = "bottom-right",
        color: tuple = None,
    ) -> Image.Image:
        """绘制丝带装饰"""
        if color is None:
            color = (255, 100, 100)  # 红色

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        ribbon_h = h // 6
        ribbon_w = w // 3

        x, y = cls._get_position(position, (w, h), (ribbon_w, ribbon_h))

        # 绘制丝带主体
        draw.polygon(
            [
                (x, y),
                (x + ribbon_w, y),
                (x + ribbon_w - ribbon_h // 3, y + ribbon_h // 2),
                (x + ribbon_w, y + ribbon_h),
                (x, y + ribbon_h),
                (x + ribbon_h // 3, y + ribbon_h // 2),
            ],
            fill=(*color, 200),
        )

        return Image.alpha_composite(img, overlay).convert("RGB")

    @classmethod
    def draw_circle(
        cls,
        image: Image.Image,
        position: str = "top-right",
        size: str = "medium",
        color: tuple = None,
    ) -> Image.Image:
        """绘制圆形装饰"""
        if color is None:
            color = (255, 215, 0, 128)  # 半透明金色

        base_size = 50
        actual_size = cls._get_size(size, base_size)

        if image.mode != "RGBA":
            img = image.convert("RGBA")
        else:
            img = image.copy()

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size

        x, y = cls._get_position(position, (w, h), (actual_size * 2, actual_size * 2))

        # 外圈
        draw.ellipse(
            [x, y, x + actual_size, y + actual_size],
            fill=(*color[:3], 100),
        )
        # 内圈
        inner = actual_size // 3
        draw.ellipse(
            [
                x + actual_size // 2 - inner // 2,
                y + actual_size // 2 - inner // 2,
                x + actual_size // 2 + inner // 2,
                y + actual_size // 2 + inner // 2,
            ],
            fill=(*color[:3], 180),
        )

        return Image.alpha_composite(img, overlay).convert("RGB")

    @classmethod
    def render_elements(
        cls,
        image: Image.Image,
        elements: list,
    ) -> Image.Image:
        """
        批量渲染装饰元素

        Args:
            image: 背景图像
            elements: 装饰元素列表，每个元素是dict:
                {
                    "type": "sparkle",  # 类型
                    "position": "top-right",  # 位置
                    "count": 3,  # 数量
                    "size": "small",  # 大小
                    "color": (255, 215, 0),  # 颜色
                    "text": "HOT",  # 文字(badge/price_tag)
                    "style": "rounded_rect"  # 样式
                }

        Returns:
            渲染后的图像
        """
        result = image.copy()

        for elem in elements:
            elem_type = elem.get("type", "sparkle")
            position = elem.get("position", "top-right")
            count = elem.get("count", 1)
            size = elem.get("size", "medium")
            color = elem.get("color")
            text = elem.get("text", "")
            style = elem.get("style", "rounded_rect")

            result = cls.render_decoration(
                image=result,
                decoration_type=elem_type,
                position=position,
                count=count,
                size=size,
                color=color,
                text=text,
                style=style,
            )

        return result

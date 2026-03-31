"""
Re-Value Agent - Node 2C: 创意合成节点
负责将装饰元素和文字渲染到最终图像上
"""

import logging
from typing import Any, Dict, Optional

from PIL import Image

from .context import Context
from tools.text_renderer import TextRenderer, render_title_on_image
from tools.decorations import DecorationRenderer

logger = logging.getLogger(__name__)


class NodeCreative:
    """
    Node 2C: 创意合成节点

    职责:
    1. 解析 VLM 输出的创意元素
    2. 使用 PIL 绘制装饰元素 (sparkles, badges, etc.)
    3. 渲染文字（从title凝练）到底部居中
    4. 输出最终合成图
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.creative_config = config.get("creative", {})
        self._text_renderer = None
        self._deco_renderer = None

    @property
    def text_renderer(self) -> TextRenderer:
        """延迟初始化文字渲染器"""
        if self._text_renderer is None:
            font_path = self.creative_config.get("font_path")
            default_size = self.creative_config.get("default_font_size", 48)
            self._text_renderer = TextRenderer(
                font_path=font_path,
                default_font_size=default_size,
            )
        return self._text_renderer

    @property
    def deco_renderer(self) -> DecorationRenderer:
        """延迟初始化装饰渲染器"""
        if self._deco_renderer is None:
            self._deco_renderer = DecorationRenderer()
        return self._deco_renderer

    async def process(self, ctx: Context) -> Context:
        """
        处理创意合成

        Args:
            ctx: 输入上下文 (包含 final_image 和 final_copy)

        Returns:
            更新后的上下文 (包含 decorated_image)
        """
        try:
            if ctx.final_image is None:
                logger.warning("NodeCreative: final_image is None, skipping")
                ctx.decorated_image = None
                return ctx

            # 从 final_copy 凝练标题文字
            title = self._extract_title(ctx.final_copy)

            # 获取装饰元素列表
            elements = ctx.creative_elements or []

            # 获取文字放置配置
            text_placement = ctx.text_placement or {"region": "bottom_center", "max_width_ratio": 0.8}

            # Step 1: 渲染装饰元素
            decorated = await self._render_decorations(ctx.final_image, elements)

            # Step 2: 渲染文字
            decorated = self._render_text(decorated, title, text_placement)

            ctx.decorated_image = decorated

            logger.info(
                "NodeCreative completed: elements=%d, title='%s...'",
                len(elements),
                title[:20] if title else "",
            )

        except Exception as e:
            logger.error("NodeCreative failed: %s", e)
            # 出错时返回原图
            ctx.decorated_image = ctx.final_image
            ctx.error_message = f"NodeCreative error: {e}"

        return ctx

    def _extract_title(self, final_copy: Optional[Dict[str, str]]) -> str:
        """
        从 final_copy 中提取/凝练标题

        Args:
            final_copy: 文案字典 {title, content}

        Returns:
            凝练后的标题
        """
        if not final_copy:
            return ""

        title = final_copy.get("title", "")

        # 凝练标题：移除emoji前缀，保留核心文字
        import re
        # 移除开头的emoji
        title = re.sub(r'^[\s]*[^\w\s]+[\s]*', '', title)
        # 移除特殊符号结尾的省略号
        title = re.sub(r'[\s…]+$', '', title)

        return title.strip()

    async def _render_decorations(
        self,
        image: Image.Image,
        elements: list,
    ) -> Image.Image:
        """
        渲染装饰元素

        Args:
            image: 背景图像
            elements: 装饰元素列表

        Returns:
            渲染后的图像
        """
        if not elements:
            return image

        try:
            return DecorationRenderer.render_elements(image, elements)
        except Exception as e:
            logger.warning("Decoration rendering failed: %s", e)
            return image

    def _render_text(
        self,
        image: Image.Image,
        title: str,
        text_placement: Dict[str, Any],
    ) -> Image.Image:
        """
        渲染文字到图像

        Args:
            image: 背景图像
            title: 标题文字
            text_placement: 放置配置

        Returns:
            渲染后的图像
        """
        if not title:
            return image

        try:
            region = text_placement.get("region", "bottom_center")
            max_width_ratio = text_placement.get("max_width_ratio", 0.8)

            # 使用文字渲染器
            return self.text_renderer.render(
                image=image,
                text=title,
                position=region,
                max_width_ratio=max_width_ratio,
                font_size=self.creative_config.get("default_font_size", 48),
                background_color=tuple(self.creative_config.get("default_text_bg", [0, 0, 0, 180])),
                text_color=tuple(self.creative_config.get("default_text_color", [255, 255, 255])),
            )
        except Exception as e:
            logger.warning("Text rendering failed: %s", e)
            return image

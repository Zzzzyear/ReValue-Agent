"""
Re-Value Agent - Node 2A: 视觉流节点
串联抠图、生图并执行 OpenCV/PIL 图像融合
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from PIL import Image

from .context import Context
from .prompts import get_sd_prompt_suffix
from tools.cv_utils import ImageFusion, MattingTool, ShadowRenderer, resize_if_needed
from tools.sd_local import get_sd_generator

logger = logging.getLogger(__name__)


class NodeVision:
    """
    Node 2A: 视觉流处理

    流程:
    Step 1: 目标分离 (Matting) - rembg 抠图
    Step 2: 背景生成 (Background Generation) - SD 生图
    Step 3: 图像融合与渲染 (Fusion & Rendering) - PIL/OpenCV 合成
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matting_config = config.get("matting", {})
        self.fusion_config = config.get("fusion", {})
        self.sd_config = config.get("stable_diffusion", {})

        # 初始化工具
        self._matting_tool = None
        self._sd_generator = None
        self._shadow_renderer = None
        self._image_fusion = None

    @property
    def matting_tool(self) -> MattingTool:
        """延迟初始化抠图工具"""
        if self._matting_tool is None:
            model_name = self.matting_config.get("model_name", "u2net")
            self._matting_tool = MattingTool(model_name=model_name)
        return self._matting_tool

    @property
    def sd_generator(self):
        """延迟初始化 SD 生成器"""
        if self._sd_generator is None:
            self._sd_generator = get_sd_generator(self.config)
        return self._sd_generator

    @property
    def shadow_renderer(self) -> ShadowRenderer:
        """延迟初始化阴影渲染器"""
        if self._shadow_renderer is None:
            self._shadow_renderer = ShadowRenderer(
                blur_radius=self.fusion_config.get("shadow_blur_radius", 15),
                shadow_alpha=self.fusion_config.get("shadow_alpha", 100),
                offset_y=self.fusion_config.get("shadow_offset_y", 10),
            )
        return self._shadow_renderer

    @property
    def image_fusion(self) -> ImageFusion:
        """延迟初始化图像融合器"""
        if self._image_fusion is None:
            self._image_fusion = ImageFusion(
                subject_height_ratio=self.fusion_config.get("subject_height_ratio", 0.65),
                bottom_margin_ratio=self.fusion_config.get("bottom_margin_ratio", 0.15),
            )
        return self._image_fusion

    async def process(self, ctx: Context) -> Context:
        """
        处理视觉流

        Args:
            ctx: 输入上下文 (包含 original_image 和 vlm_json_output)

        Returns:
            更新后的上下文 (包含 matted_image_rgba, background_image, final_image)
        """
        try:
            # 获取原始图像
            image = ctx.original_image
            if image is None:
                raise ValueError("No original image in context")

            # Step 1: 抠图
            logger.info("Step 1: Starting matting")
            ctx.matted_image_rgba = await self._process_matting(image)
            logger.info("Step 1: Matting completed")

            # Step 2: 背景生成 (与 Step 1 并行，但这里简化处理)
            logger.info("Step 2: Starting background generation")
            background_prompt = ctx.background_prompt + get_sd_prompt_suffix()
            ctx.background_image = await self._process_background(background_prompt)
            logger.info("Step 2: Background generation completed")

            # Step 3: 图像融合
            logger.info("Step 3: Starting image fusion")
            ctx.final_image = self._process_fusion(
                subject_rgba=ctx.matted_image_rgba,
                background=ctx.background_image,
                reference_box=ctx.reference_box,
            )
            logger.info("Step 3: Fusion completed")

        except Exception as e:
            logger.error("NodeVision process failed: %s", e)
            # 视觉流失败不影响整体，使用原图作为 fallback
            ctx.final_image = ctx.original_image
            ctx.error_message = f"Vision error: {e}"

        return ctx

    async def _process_matting(self, image: Image.Image) -> Image.Image:
        """
        执行抠图

        Args:
            image: 原始 RGB 图像

        Returns:
            抠图后的 RGBA 图像
        """
        # 限制图像大小，避免 rembg 处理超大图像
        max_size = 1024
        resized = False
        original_size = image.size

        if max(image.size) > max_size:
            image = resize_if_needed(image, max_size=max_size)
            resized = True
            logger.info("Image resized to %s for matting", image.size)

        # 在 executor 中运行以避免阻塞
        loop = asyncio.get_event_loop()
        matted = await loop.run_in_executor(
            None,
            lambda: self.matting_tool.remove_background(image)
        )

        # 如果之前缩放过，恢复原始尺寸
        if resized:
            matted = matted.resize(original_size, Image.LANCZOS)

        return matted

    async def _process_background(self, prompt: str) -> Image.Image:
        """
        执行背景生成

        Args:
            prompt: SD 生图提示词

        Returns:
            生成的背景图
        """
        width = self.sd_config.get("width", 1024)
        height = self.sd_config.get("height", 1024)
        num_inference_steps = self.sd_config.get("num_inference_steps", 25)
        guidance_scale = self.sd_config.get("guidance_scale", 7.5)

        return await self.sd_generator.generate(
            prompt=prompt,
            negative_prompt="product on ground, messy background, low quality",
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def _process_fusion(
        self,
        subject_rgba: Image.Image,
        background: Image.Image,
        reference_box: Optional[list],
    ) -> Image.Image:
        """
        执行图像融合

        Args:
            subject_rgba: 抠图后的 RGBA 图像
            background: 背景图
            reference_box: 参考框 (可选)

        Returns:
            合成后的图像
        """
        # 使用融合器进行合成
        fused = self.image_fusion.fuse(
            subject_rgba=subject_rgba,
            background=background,
            reference_box=reference_box,
            shadow_renderer=self.shadow_renderer,
        )

        return fused

    async def process_matting_only(self, image: Image.Image) -> Image.Image:
        """
        仅执行抠图 (用于测试或独立调用)

        Args:
            image: 输入图像

        Returns:
            抠图后的 RGBA 图像
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.matting_tool.remove_background(image)
        )

    async def process_background_only(self, prompt: str) -> Image.Image:
        """
        仅执行背景生成 (用于测试或独立调用)

        Args:
            prompt: 生图提示词

        Returns:
            生成的背景图
        """
        return await self.sd_generator.generate(prompt=prompt)

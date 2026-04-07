"""
Re-Value Agent - 本地 Stable Diffusion 推理封装
支持 SD 1.5 / SDXL-Turbo / SDXL 等模型
提供异步推理能力，不阻塞事件循环
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class BaseSDGenerator(ABC):
    """SD 生成器抽象基类"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        生成图像

        Args:
            prompt: 生图提示词
            negative_prompt: 负面提示词
            width: 图像宽度
            height: 图像高度
            num_inference_steps: 推理步数
            guidance_scale: CFG 引导强度

        Returns:
            生成的 PIL Image
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass


class DummySDGenerator(BaseSDGenerator):
    """
    Dummy SD 生成器 (用于测试或无GPU环境)
    返回纯色渐变图作为占位符
    """

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """返回渐变色占位图"""
        logger.warning("Using dummy SD generator (no actual image generation)")

        # 创建渐变色占位图
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        for y in range(height):
            for x in range(width):
                r = int(240 - (y / height) * 40)
                g = int(248 - (y / height) * 30)
                b = int(255 - (y / height) * 20)
                pixels[x, y] = (r, g, b)

        return img

    def is_available(self) -> bool:
        return True


class DiffusersSDGenerator(BaseSDGenerator):
    """
    基于 diffusers 库的 Stable Diffusion 生成器
    支持 SD 1.5 / SDXL / SDXL-Turbo
    """

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        model_type: str = "sd_15",  # sd_15 | sdxl | sdxl_turbo
        device: str = "cuda",
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        disable_safety_checker: bool = False,
    ):
        """
        Args:
            model_path: 模型路径或 HuggingFace repo_id
            model_type: 模型类型
            device: 推理设备 (cuda / cpu)
            enable_attention_slicing: 启用注意力切片 (省显存)
            enable_vae_slicing: 启用 VAE 切片 (省显存)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_slicing = enable_vae_slicing
        self.disable_safety_checker = disable_safety_checker

        self._pipeline = None
        self._lock = asyncio.Lock()  # 防止并发加载模型

    def _get_default_negative_prompt(self) -> str:
        """获取默认负面提示词"""
        return (
            "blurry, low quality, watermark, text, logo, signature, "
            "cropped, worst quality, low resolution, monochrome, "
            "graffiti, illustration, painting, drawing, anime style"
        )

    def _load_pipeline(self):
        """加载 SD Pipeline (同步方法，应在 executor 中调用)"""
        if self._pipeline is not None:
            return

        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                DiffusionPipeline,
                DPMSolverMultistepScheduler,
                UNet2DConditionModel,
            )

            logger.info("Loading SD pipeline: %s (type: %s)", self.model_path, self.model_type)

            if self.model_type == "sdxl_turbo":
                # SDXL-Turbo (快速出图，1-4步)
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16" if self.device == "cuda" else None,
                )
            elif self.model_type == "sdxl":
                # SDXL 原生
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16" if self.device == "cuda" else None,
                )
            else:
                # SD 1.5
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )

            # 禁用 NSFW safety checker（避免误检导致返回黑图）
            if self.disable_safety_checker:
                pipe.safety_checker = None
                pipe.requires_safety_checker = False
                logger.info("NSFW safety checker disabled")

            # 显存优化
            if self.enable_attention_slicing:
                pipe.enable_attention_slicing()
            if self.enable_vae_slicing:
                pipe.enable_vae_slicing()

            # 移动到设备
            pipe = pipe.to(self.device)

            # 优化调度器
            if self.model_type == "sdxl_turbo":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )

            self._pipeline = pipe
            logger.info("SD pipeline loaded successfully")

        except ImportError as e:
            logger.error("Failed to import diffusers: %s", e)
            raise ImportError(
                "Please install diffusers: pip install diffusers transformers accelerate torch"
            )

    async def _generate_async(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        """异步生成 (在后台线程运行以避免阻塞)"""
        loop = asyncio.get_event_loop()

        def _generate():
            if self._pipeline is None:
                self._load_pipeline()

            # SDXL-Turbo 快速模式
            steps = num_inference_steps
            if self.model_type == "sdxl_turbo":
                steps = min(steps, 4)

            result = self._pipeline(
                prompt,
                negative_prompt=negative_prompt or self._get_default_negative_prompt(),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            return result.images[0]

        return await loop.run_in_executor(None, _generate)

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """异步生成图像"""
        async with self._lock:
            return await self._generate_async(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

    def is_available(self) -> bool:
        """检查模型是否可用"""
        if self._pipeline is not None:
            return True

        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available")
                return False
            return True
        except ImportError:
            return False


# =============================================================================
# SD 生成器工厂
# =============================================================================

def get_sd_generator(config: Dict[str, Any]) -> BaseSDGenerator:
    """
    获取 SD 生成器实例

    Args:
        config: 全局配置

    Returns:
        SD 生成器实例
    """
    import torch

    sd_config = config.get("stable_diffusion", {})
    model_type = sd_config.get("model_type", "sd_15")
    model_path = sd_config.get("model_path", "runwayml/stable-diffusion-v1-5")

    # 检查是否有 GPU
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    if not has_cuda:
        logger.warning(
            "No CUDA GPU available, using dummy SD generator. "
            "Image generation will produce placeholder images."
        )
        return DummySDGenerator()

    # 根据配置创建生成器
    generator = DiffusersSDGenerator(
        model_path=model_path,
        model_type=model_type,
        device=device,
        enable_attention_slicing=sd_config.get("enable_attention_slicing", True),
        enable_vae_slicing=sd_config.get("enable_vae_slicing", True),
        disable_safety_checker=sd_config.get("disable_safety_checker", False),
    )

    logger.info(
        "SD Generator initialized: type=%s, path=%s, device=%s",
        model_type,
        model_path,
        device,
    )

    return generator


def check_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_memory_info() -> Dict[str, float]:
    """获取 GPU 显存信息 (GB)"""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            return {"total": round(total, 2), "allocated": round(allocated, 2)}
    except ImportError:
        pass
    return {"total": 0, "allocated": 0}

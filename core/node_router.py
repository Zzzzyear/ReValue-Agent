"""
Re-Value Agent - Node 1: VLM 联合感知与路由分发
调用 VLM，解析 JSON，处理坐标与路由下发
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional
from .prompts import FALLBACK_JSON, get_vlm_prompts, parse_json_from_response

logger = logging.getLogger(__name__)


class NodeRouter:
    """
    Node 1: VLM 联合感知与路由分发

    职责:
    1. 接收原始图片和文本
    2. 调用 VLM API 提取商品信息
    3. 解析 JSON 输出
    4. 下发路由指令到后续节点
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vlm_config = config.get("vlm", {})
        self.api_client = None  # 延迟初始化

        # 支持的 VLM 模型
        self.supported_models = ["qwen_vl", "glm_4v", "kimi_vl"]

    async def process(self, ctx: Context) -> Context:
        """
        处理输入，调用 VLM 并解析结果

        Args:
            ctx: 输入上下文

        Returns:
            更新后的上下文
        """
        try:
            # 获取 Prompt
            prompts = get_vlm_prompts(ctx.original_text)

            # 将图片转换为 base64
            from .engine import ReValueEngine
            image_base64 = ReValueEngine.image_to_base64(ctx.original_image)

            # 调用 VLM API
            response = await self._call_vlm(
                image_base64=image_base64,
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
            )

            # 解析 JSON 响应
            vlm_result = parse_json_from_response(response)

            # 更新上下文
            ctx.vlm_json_output = vlm_result
            ctx.subject = vlm_result.get("subject", "商品")
            ctx.selling_points = vlm_result.get("selling_points", [])
            ctx.background_prompt = vlm_result.get("background_prompt", "")
            ctx.reference_box = vlm_result.get("reference_box")
            ctx.creative_elements = vlm_result.get("creative_elements", [])
            ctx.text_placement = vlm_result.get("text_placement", {"region": "bottom_center", "max_width_ratio": 0.8})

            logger.info(
                "Node 1 completed: subject=%s, points=%d, has_box=%s, elements=%d",
                ctx.subject,
                len(ctx.selling_points),
                ctx.reference_box is not None,
                len(ctx.creative_elements) if ctx.creative_elements else 0,
            )

        except Exception as e:
            logger.error("Node 1 failed: %s", e)

            # 使用 Fallback JSON
            ctx.vlm_json_output = FALLBACK_JSON
            ctx.subject = FALLBACK_JSON["subject"]
            ctx.selling_points = FALLBACK_JSON["selling_points"]
            ctx.background_prompt = FALLBACK_JSON["background_prompt"]
            ctx.reference_box = FALLBACK_JSON["reference_box"]
            ctx.creative_elements = FALLBACK_JSON.get("creative_elements", [])
            ctx.text_placement = FALLBACK_JSON.get("text_placement", {"region": "bottom_center", "max_width_ratio": 0.8})

            ctx.error_message = f"Node 1 error (using fallback): {e}"

        return ctx

    async def _call_vlm(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        调用 VLM API

        这里预留接口，支持切换不同的 VLM 提供商

        Args:
            image_base64: base64 编码的图片
            system_prompt: 系统提示
            user_prompt: 用户提示

        Returns:
            VLM 响应文本
        """
        model = self.vlm_config.get("default_model", "qwen_vl")

        if self.api_client is None:
            # 延迟初始化 API 客户端
            from tools.api_clients import get_vlm_client
            self.api_client = get_vlm_client(model, self.config)

        return await self.api_client.chat_with_image(
            image_base64=image_base64,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def get_api_client(self):
        """获取当前 API 客户端（用于外部调用）"""
        if self.api_client is None:
            model = self.vlm_config.get("default_model", "qwen_vl")
            from tools.api_clients import get_vlm_client
            self.api_client = get_vlm_client(model, self.config)
        return self.api_client

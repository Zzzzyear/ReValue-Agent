"""
Re-Value Agent - Node 2B: 文本流节点
组装文案 Prompt 并调用大语言模型生成营销文案
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from .context import Context
from .prompts import get_copy_prompts
from tools.api_clients import get_llm_client

logger = logging.getLogger(__name__)


class NodeText:
    """
    Node 2B: 文案生成节点

    职责:
    1. 组装文案 Prompt
    2. 调用 LLM API 生成文案
    3. 解析并返回结构化文案
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.platform_config = config.get("platform_style", {})
        self._llm_client = None

    @property
    def llm_client(self):
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None:
            model = self.llm_config.get("default_model", "qwen_max")
            self._llm_client = get_llm_client(model, self.config)
        return self._llm_client

    async def process(self, ctx: Context) -> Context:
        """
        处理文案生成

        Args:
            ctx: 输入上下文 (包含 subject, selling_points, original_text)

        Returns:
            更新后的上下文 (包含 final_copy)
        """
        try:
            # 获取平台风格（优先使用 Context 中的设置）
            platform_style = getattr(ctx, 'platform_style', None) or self.platform_config.get("default", "闲鱼体")

            # 组装 Prompt
            prompts = get_copy_prompts(
                subject=ctx.subject,
                selling_points=ctx.selling_points,
                original_text=ctx.original_text,
                platform_style=platform_style,
            )

            # 调用 LLM
            response = await self._call_llm(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
            )

            # 解析响应
            ctx.final_copy = self._parse_copy_response(response)

            logger.info(
                "NodeText completed: title=%s",
                ctx.final_copy.get("title", "N/A")[:30] if ctx.final_copy else "N/A",
            )

        except Exception as e:
            logger.error("NodeText process failed: %s", e)

            # Fallback: 使用原始文本
            ctx.final_copy = {
                "title": ctx.subject or "二手好物",
                "content": ctx.original_text,
            }
            ctx.error_message = f"Text error: {e}"

        return ctx

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用 LLM API

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示

        Returns:
            LLM 响应文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.llm_client.chat(messages)

    def _parse_copy_response(self, response: str) -> Dict[str, str]:
        """
        解析 LLM 响应，提取文案

        Args:
            response: LLM 原始响应

        Returns:
            {"title": "...", "content": "..."}
        """
        # 策略1: 直接解析 JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 策略2: 提取 ```json ... ``` 块
        json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        # 策略3: 提取 {...} 内容
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 策略4: 手动解析 key-value
        result = {}

        title_match = re.search(r'"title"\s*:\s*"([^"]*)"', response)
        if title_match:
            result["title"] = title_match.group(1)

        content_match = re.search(r'"content"\s*:\s*"([^"]*)"', response, re.DOTALL)
        if content_match:
            result["content"] = content_match.group(1).replace('\\n', '\n')

        if result:
            return result

        # 所有策略都失败，返回默认值
        logger.warning("Failed to parse copy response, using fallback")
        return {
            "title": "二手好物转让",
            "content": response[:200] if response else "优质二手商品，欢迎咨询",
        }

    async def generate_copy_only(
        self,
        subject: str,
        selling_points: List[str],
        original_text: str,
        platform_style: str = "闲鱼体",
    ) -> Dict[str, str]:
        """
        仅生成文案 (用于测试或独立调用)

        Args:
            subject: 商品名称
            selling_points: 卖点列表
            original_text: 原始描述
            platform_style: 平台风格

        Returns:
            {"title": "...", "content": "..."}
        """
        ctx = Context(
            subject=subject,
            selling_points=selling_points,
            original_text=original_text,
        )

        ctx = await self.process(ctx)
        return ctx.final_copy or {"title": subject, "content": original_text}

"""
ReValue-Agent - 统一的 API 请求客户端
封装 tenacity 重试机制与 Semaphore 并发锁
支持: Qwen-VL, GLM-4V, KIMI VL, Qwen-Max, GLM-4 等
"""

import asyncio
import base64
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# API Key 配置加载
# =============================================================================

def _get_shared_api_config() -> Dict[str, Any]:
    """
    从共享目录加载 API 配置

    配置优先级（从高到低）：
    1. 环境变量（生产环境）
    2. shared/api_keys_local.yaml（本地覆盖）
    3. shared/api_keys.yaml（模板配置）

    Returns:
        API 配置字典
    """
    import yaml

    # 共享配置目录（向上两级到 AI_Playground，再进入 shared）
    shared_dir = Path(__file__).parent.parent.parent.parent / "shared"

    # 优先加载本地覆盖配置
    local_config_path = shared_dir / "api_keys_local.yaml"
    if local_config_path.exists():
        try:
            with open(local_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config and config.get("dashscope", {}).get("api_key"):
                    logger.info("Using local API config from api_keys_local.yaml")
                    return config
        except Exception as e:
            logger.warning(f"Failed to load local API config: {e}")

    # 加载模板配置
    template_config_path = shared_dir / "api_keys.yaml"
    if template_config_path.exists():
        try:
            with open(template_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    logger.info("Using template API config from api_keys.yaml")
                    return config
        except Exception as e:
            logger.warning(f"Failed to load template API config: {e}")

    # 返回空配置，使用环境变量
    logger.warning("No shared API config found, using environment variables")
    return {}


def _get_api_key(provider: str) -> str:
    """
    获取 API Key，优先从共享配置读取

    Args:
        provider: 提供商名称 (dashscope, glm, kimi)

    Returns:
        API Key 字符串
    """
    # 1. 尝试从共享配置获取
    shared_config = _get_shared_api_config()

    if shared_config:
        # DashScope (Qwen)
        if provider == "dashscope":
            key = shared_config.get("dashscope", {}).get("api_key", "")
            if key and key != "YOUR_API_KEY_HERE":
                return key

        # GLM
        if provider == "glm":
            key = shared_config.get("glm", {}).get("api_key", "")
            if key:
                return key

        # KIMI
        if provider == "kimi":
            key = shared_config.get("kimi", {}).get("api_key", "")
            if key:
                return key

    # 2. 回退到环境变量
    env_map = {
        "dashscope": ["DASHSCOPE_API_KEY", "QWEN_VL_API_KEY", "QWEN_MAX_API_KEY"],
        "glm": ["GLM_4V_API_KEY", "GLM_API_KEY"],
        "kimi": ["KIMI_API_KEY"],
    }

    for env_var in env_map.get(provider, []):
        key = os.getenv(env_var, "")
        if key:
            return key

    return ""


# =============================================================================
# VLM / LLM 抽象接口
# =============================================================================

class BaseAPIClient(ABC):
    """API 客户端抽象基类"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp Session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """关闭 Session"""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def chat(
        self,
        messages: list,
        **kwargs,
    ) -> str:
        """发送聊天请求，返回响应文本"""
        pass

    @abstractmethod
    async def chat_with_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """发送带图片的聊天请求（VLM 专用）"""
        pass


# =============================================================================
# Qwen-VL / Qwen-Max 客户端
# =============================================================================

class QwenClient(BaseAPIClient):
    """阿里 Qwen VLM/LLM 客户端 (OpenAI-compatible API)"""

    async def chat(
        self,
        messages: list,
        **kwargs,
    ) -> str:
        session = await self._get_session()
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        async def _do_request():
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        message=f"API error {resp.status}: {text}",
                    )
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        return await self._retry_request(_do_request)

    async def chat_with_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """Qwen-VL 支持图片输入"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return await self.chat(messages, **kwargs)

    async def _retry_request(self, fn):
        """使用 tenacity 重试"""
        wait = wait_exponential(multiplier=2, min=2, max=10)
        stop = stop_after_attempt(self.max_retries)

        retrying = retry(wait=wait, stop=stop, reraise=True)

        async def _wrapped():
            attempt = 0
            while True:
                attempt += 1
                try:
                    return await fn()
                except Exception as e:
                    if attempt >= self.max_retries:
                        raise
                    logger.warning(
                        "Request failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        e,
                    )
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)

        return await _wrapped()


# =============================================================================
# GLM-4V / GLM-4 客户端
# =============================================================================

class GLMClient(BaseAPIClient):
    """智谱 GLM-4V/GLM-4 客户端"""

    async def chat(
        self,
        messages: list,
        **kwargs,
    ) -> str:
        session = await self._get_session()
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        async def _do_request():
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        message=f"API error {resp.status}: {text}",
                    )
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        return await self._retry_request(_do_request)

    async def chat_with_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """GLM-4V 图片输入"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return await self.chat(messages, **kwargs)

    async def _retry_request(self, fn):
        """重试逻辑"""
        async def _wrapped():
            for attempt in range(1, self.max_retries + 1):
                try:
                    return await fn()
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    logger.warning(
                        "Request failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        e,
                    )
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)

        return await _wrapped()


# =============================================================================
# KIMI VL 客户端
# =============================================================================

class KIMIClient(BaseAPIClient):
    """月之暗面 KIMI VL 客户端 (OpenAI-compatible)"""

    async def chat(
        self,
        messages: list,
        **kwargs,
    ) -> str:
        session = await self._get_session()
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        async def _do_request():
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        message=f"API error {resp.status}: {text}",
                    )
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        return await self._retry_request(_do_request)

    async def chat_with_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return await self.chat(messages, **kwargs)

    async def _retry_request(self, fn):
        async def _wrapped():
            for attempt in range(1, self.max_retries + 1):
                try:
                    return await fn()
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    logger.warning(
                        "Request failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        e,
                    )
                    wait_time = min(2 ** attempt, 10)
                    await asyncio.sleep(wait_time)

        return await _wrapped()


# =============================================================================
# 客户端工厂函数
# =============================================================================

def get_vlm_client(model: str, config: Dict[str, Any]) -> BaseAPIClient:
    """
    获取 VLM 客户端

    Args:
        model: 模型名称 (qwen_vl, glm_4v, kimi_vl)
        config: 全局配置

    Returns:
        对应的 API 客户端实例
    """
    vlm_config = config.get("vlm", {})
    api_base = vlm_config.get("api_base", "https://api.example.com/v1")
    timeout = vlm_config.get("timeout", 30)
    max_retries = config.get("concurrency", {}).get("max_retries", 3)

    if "qwen" in model.lower():
        api_key = _get_api_key("dashscope")
        return QwenClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif "glm" in model.lower():
        api_key = _get_api_key("glm")
        return GLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif "kimi" in model.lower():
        api_key = _get_api_key("kimi")
        return KIMIClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        # 默认使用 Qwen
        api_key = _get_api_key("dashscope")
        return QwenClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )


def get_llm_client(model: str, config: Dict[str, Any]) -> BaseAPIClient:
    """
    获取 LLM 客户端 (纯文本，无图像支持)

    Args:
        model: 模型名称 (qwen_max, glm_4, kimi)
        config: 全局配置

    Returns:
        对应的 API 客户端实例
    """
    llm_config = config.get("llm", {})
    api_base = llm_config.get("api_base", "https://api.example.com/v1")
    timeout = llm_config.get("timeout", 30)
    max_retries = config.get("concurrency", {}).get("max_retries", 3)

    if "qwen" in model.lower():
        api_key = _get_api_key("dashscope")
        return QwenClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif "glm" in model.lower():
        api_key = _get_api_key("glm")
        return GLMClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif "kimi" in model.lower():
        api_key = _get_api_key("kimi")
        return KIMIClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        api_key = _get_api_key("dashscope")
        return QwenClient(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

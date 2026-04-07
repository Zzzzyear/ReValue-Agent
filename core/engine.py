"""
Re-Value Agent - 核心调度引擎
基于 asyncio 的 DAG 任务组装与汇聚逻辑
"""

import asyncio
import base64
import io
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .context import Context, TaskStatus
from .node_router import NodeRouter
from .node_vision import NodeVision
from .node_text import NodeText
from .node_creative import NodeCreative

logger = logging.getLogger(__name__)


class ReValueEngine:
    """
    Re-Value Agent 核心调度引擎

    DAG 流程:
        Node 1 (Router) ─┬─→ Node 2A (Vision) ─┐
                         │                      ├→ Node 2C (Creative) → Output
                         └─→ Node 2B (Text)  ────┘
    """

    def __init__(
        self,
        config: Dict[str, Any],
        semaphore_limit: int = 2,
    ):
        self.config = config
        self.semaphore = asyncio.Semaphore(semaphore_limit)

        # 初始化节点
        self.node_router = NodeRouter(config)
        self.node_vision = NodeVision(config)
        self.node_text = NodeText(config)
        self.node_creative = NodeCreative(config)

        logger.info("ReValueEngine initialized with semaphore limit: %d", semaphore_limit)

    async def process(
        self,
        image: Image.Image,
        text: str,
        platform_style: str = "闲鱼体",
    ) -> Context:
        """
        主入口: 处理单张图片和文本

        Args:
            image: PIL Image 对象
            text: 原始描述文本

        Returns:
            Context: 包含所有处理结果的上下文
        """
        ctx = Context(
            original_image=image,
            original_text=text,
            platform_style=platform_style,
            status=TaskStatus.RUNNING,
        )

        try:
            # Node 1: VLM 联合感知与路由分发
            logger.info("Starting Node 1: VLM Router")
            ctx = await self.node_router.process(ctx)

            if not ctx.subject:
                raise ValueError("Node 1 failed: No subject extracted from VLM")

            # Node 2: 异步并行处理 (视觉流 + 文本流)
            logger.info("Starting Node 2: Parallel Vision + Text")

            vision_task = asyncio.create_task(self._process_vision_branch(ctx))
            text_task = asyncio.create_task(self._process_text_branch(ctx))

            # 等待双路完成
            vision_result, text_result = await asyncio.gather(
                vision_task,
                text_task,
                return_exceptions=True,
            )

            # 处理异常
            if isinstance(vision_result, Exception):
                logger.error("Vision branch failed: %s", vision_result)
                ctx.error_message = f"Vision error: {vision_result}"

            if isinstance(text_result, Exception):
                logger.error("Text branch failed: %s", text_result)
                # 文案失败不影响整体，使用fallback文案
                ctx.final_copy = {
                    "title": ctx.subject,
                    "content": ctx.original_text,
                }

            # Node 2C: 创意合成 (装饰元素 + 文字渲染)
            logger.info("Starting Node 2C: Creative Composition")
            ctx = await self.node_creative.process(ctx)

            # Node 3: 结果汇聚
            ctx.status = TaskStatus.COMPLETED
            logger.info("Processing completed successfully")

        except Exception as e:
            logger.error("Engine process failed: %s", e)
            ctx.status = TaskStatus.FAILED
            ctx.error_message = str(e)
        finally:
            await self.close()

        return ctx

    async def close(self):
        """关闭所有 API 客户端 session，释放资源"""
        for node in [self.node_router, self.node_text]:
            client = getattr(node, 'api_client', None) or getattr(node, '_llm_client', None)
            if client and hasattr(client, 'close'):
                try:
                    await client.close()
                except Exception as e:
                    logger.warning("Failed to close client: %s", e)

    async def _process_vision_branch(self, ctx: Context) -> Context:
        """视觉流处理 (Branch A)"""
        async with self.semaphore:
            return await self.node_vision.process(ctx)

    async def _process_text_branch(self, ctx: Context) -> Context:
        """文本流处理 (Branch B)"""
        async with self.semaphore:
            return await self.node_text.process(ctx)

    async def process_batch(
        self,
        items: List[Tuple[Image.Image, str]],
    ) -> List[Context]:
        """
        批量处理多张图片

        Args:
            items: List of (image, text) tuples

        Returns:
            List of Context objects
        """
        tasks = [self.process(img, txt) for img, txt in items]
        return await asyncio.gather(*tasks)

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """将 PIL Image 转换为 Base64 字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def base64_to_image(base64_str: str) -> Image.Image:
        """将 Base64 字符串转换为 PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

    @staticmethod
    def save_output(
        ctx: Context,
        output_dir: str = "data/output",
        prefix: str = "result",
    ) -> Dict[str, str]:
        """
        保存结果到指定目录

        Returns:
            Dict with paths to saved files
        """
        import os
        import time

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        paths = {}

        # 保存最终合成图 (优先使用 decorated_image，否则使用 final_image)
        output_image = ctx.decorated_image or ctx.final_image
        if output_image:
            img_path = os.path.join(output_dir, f"{prefix}_{timestamp}_final.png")
            output_image.save(img_path)
            paths["final_image"] = img_path

            # 如果 decorated_image 和 final_image 不同，额外保存 decorated 版本
            if ctx.decorated_image and ctx.decorated_image != ctx.final_image:
                deco_path = os.path.join(output_dir, f"{prefix}_{timestamp}_decorated.png")
                ctx.decorated_image.save(deco_path)
                paths["decorated_image"] = deco_path

        # 保存中间产物 (可选)
        if ctx.matted_image_rgba:
            mat_path = os.path.join(output_dir, f"{prefix}_{timestamp}_matted.png")
            ctx.matted_image_rgba.save(mat_path)
            paths["matted_image"] = mat_path

        if ctx.background_image:
            bg_path = os.path.join(output_dir, f"{prefix}_{timestamp}_background.png")
            ctx.background_image.save(bg_path)
            paths["background"] = bg_path

        # 保存文案
        if ctx.final_copy:
            import json
            txt_path = os.path.join(output_dir, f"{prefix}_{timestamp}_copy.json")
            with open(txt_path, "w", encoding="utf-8") as f:
                json.dump(ctx.final_copy, f, ensure_ascii=False, indent=2)
            paths["copy"] = txt_path

        # 保存完整上下文 (JSON)
        import json
        ctx_path = os.path.join(output_dir, f"{prefix}_{timestamp}_context.json")
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)
        paths["context"] = ctx_path

        return paths

    @staticmethod
    def save_full_report(
        ctx: Context,
        result_paths: Dict[str, str],
        elapsed_time: float = 0.0,
        output_dir: str = "data/output",
        prefix: str = "result",
    ) -> str:
        """
        生成详细的处理报告，记录每一步的输入输出

        Args:
            ctx: 处理上下文
            result_paths: save_output 返回的文件路径字典
            elapsed_time: 总耗时（秒）
            output_dir: 输出目录
            prefix: 文件前缀

        Returns:
            报告文件路径
        """
        import json
        import time
        from pathlib import Path

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())

        # 构建完整报告
        report = {
            # ========== 任务基本信息 ==========
            "task_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "timestamp_unix": timestamp,
                "elapsed_time_seconds": round(elapsed_time, 2),
                "status": ctx.status.value,
                "error_message": ctx.error_message,
            },

            # ========== 输入信息 ==========
            "input": {
                "original_text": ctx.original_text,
                "original_image_path": ctx.original_image_path,
                # 图片尺寸信息（如果有）
                "original_image_size": ctx.original_image.size if ctx.original_image else None,
            },

            # ========== Node 1: VLM 分析结果 ==========
            "node1_vlm": {
                "subject": ctx.subject,
                "selling_points": ctx.selling_points,
                "background_prompt": ctx.background_prompt,
                "reference_box": ctx.reference_box,
                "creative_elements": ctx.creative_elements,
                "text_placement": ctx.text_placement,
                "vlm_raw_output": ctx.vlm_json_output,
            },

            # ========== Node 2A: 视觉处理结果 ==========
            "node2a_vision": {
                "has_matted_image": ctx.matted_image_rgba is not None,
                "has_background_image": ctx.background_image is not None,
                "has_final_image": ctx.final_image is not None,
                "output_paths": {
                    "matted_image": result_paths.get("matted_image", ""),
                    "background_image": result_paths.get("background", ""),
                    "final_image": result_paths.get("final_image", ""),
                },
            },

            # ========== Node 2B: 文案生成结果 ==========
            "node2b_text": {
                "final_copy": ctx.final_copy,
                "output_path": result_paths.get("copy", ""),
            },

            # ========== Node 2C: 创意合成结果 ==========
            "node2c_creative": {
                "has_decorated_image": ctx.decorated_image is not None,
                "decorated_image_path": result_paths.get("decorated_image", ""),
            },

            # ========== 最终输出文件列表 ==========
            "output_files": result_paths,
        }

        # 保存报告
        report_path = os.path.join(output_dir, f"{prefix}_{timestamp}_full_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("Full report saved to: %s", report_path)
        return report_path

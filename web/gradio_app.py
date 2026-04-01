"""
Re-Value Agent - Gradio Web UI
基于 Gradio 的极简 Demo 界面
"""

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import ReValueEngine
from tools.sd_local import check_cuda_available, get_gpu_memory_info

logger = logging.getLogger(__name__)

# 全局配置
CONFIG_PATH = Path(__file__).parent.parent / "conf" / "config.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"


def load_config():
    """加载配置"""
    import yaml

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class ReValueGradioApp:
    """
    Re-Value Agent Gradio 应用
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()

        # 检查 GPU
        self.has_cuda = check_cuda_available()
        if self.has_cuda:
            gpu_info = get_gpu_memory_info()
            logger.info("GPU available: %.1f GB total, %.1f GB allocated",
                       gpu_info["total"], gpu_info["allocated"])
        else:
            logger.warning("No GPU available, using dummy SD generator")

        # 初始化引擎
        semaphore_limit = self.config.get("concurrency", {}).get("semaphore_limit", 2)
        self.engine = ReValueEngine(self.config, semaphore_limit=semaphore_limit)

        # 输出目录
        self.output_dir = Path(self.config.get("paths", {}).get("data_output", str(DEFAULT_OUTPUT_DIR)))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 简单的并发锁 (Gradio 协程不安全)
        self._lock = threading.Lock()

        # 示例图片使用 data/original 目录下的真实图片
        project_root = Path(__file__).parent.parent
        self.examples = [
            [str(project_root / "data" / "original" / "xianyu_bag_01.jpg"), "爱马仕Kelly二手包"],
            [str(project_root / "data" / "original" / "xianyu_shoes_01.jpg"), "李宁音速11球鞋"],
            [str(project_root / "data" / "original" / "shoes_02.jpg"), "Nike足球鞋九成新"],
            [str(project_root / "data" / "original" / "xiaohongshu_bag_01.jpg"), "LV月亮包二手"],
        ]

    async def _process_async(
        self,
        image: Image.Image,
        text: str,
        platform_style: str = "闲鱼体",
        creative_type: str = "全部",
        progress=gr.Progress(),
    ) -> tuple:
        """
        异步处理单张图片

        Args:
            image: 上传的图像
            text: 商品描述
            platform_style: 平台风格 (闲鱼体/小红书体)
            creative_type: 创意装饰类型

        Returns:
            (最终图像, 文案标题, 文案内容, 状态信息)
        """
        try:
            progress(0.0, desc="正在处理...")
            start_time = time.time()

            # 阶段1: VLM 分析
            progress(0.1, desc="[1/4] VLM 正在分析商品特征...")

            # 创建 Context 并设置平台风格
            from core.context import Context, TaskStatus
            ctx = Context(
                original_image=image,
                original_text=text.strip(),
                status=TaskStatus.RUNNING,
                platform_style=platform_style,
            )

            # 处理 VLM 路由
            ctx = await self.engine.node_router.process(ctx)
            if ctx.status.value == "failed":
                return None, "", "", f"VLM分析失败: {ctx.error_message}"

            # 处理文案生成
            progress(0.2, desc="[2/4] 正在生成营销文案...")
            ctx = await self.engine.node_text.process(ctx)

            # 处理视觉流
            progress(0.3, desc="[3/4] 正在进行视觉处理...")
            ctx = await self.engine._process_vision_branch(ctx)

            # 处理创意合成
            progress(0.8, desc="[4/4] 正在渲染装饰与文字...")
            ctx = await self.engine.node_creative.process(ctx)

            ctx.status = TaskStatus.COMPLETED

            elapsed = time.time() - start_time

            if ctx.status.value == "completed":
                progress(1.0, desc="处理完成!")

                # 保存结果
                result_paths = ReValueEngine.save_output(
                    ctx,
                    output_dir=str(self.output_dir),
                    prefix="gradio",
                )

                # 生成详细报告
                report_path = ReValueEngine.save_full_report(
                    ctx,
                    result_paths,
                    elapsed_time=elapsed,
                    output_dir=str(self.output_dir),
                    prefix="gradio",
                )

                status = f"处理完成! 耗时 {elapsed:.1f}秒"
                if result_paths.get("final_image"):
                    status += f"\n已保存至: {result_paths['final_image']}"
                status += f"\n详细报告: {report_path}"

                # 返回 decorated_image (带装饰+文字的最终图)
                output_image = ctx.decorated_image or ctx.final_image
                return (
                    output_image,
                    ctx.final_copy.get("title", "") if ctx.final_copy else "",
                    ctx.final_copy.get("content", "") if ctx.final_copy else "",
                    status,
                )
            else:
                return (
                    None,
                    "",
                    "",
                    f"处理失败: {ctx.error_message}",
                )

        except Exception as e:
            logger.error("Process failed: %s", e)
            return None, "", "", f"处理异常: {str(e)}"

    def process_image(
        self,
        image: Image.Image,
        text: str,
        platform_style: str = "闲鱼体",
        creative_type: str = "全部",
        progress=gr.Progress(),
    ) -> tuple:
        """同步入口，调用异步处理"""
        if image is None:
            return None, "", "", "请上传商品图片"

        if not text or not text.strip():
            return None, "", "", "请输入商品描述"

        with self._lock:
            # 使用 try-except 包装 asyncio.run 调用
            try:
                return asyncio.run(self._process_async(
                    image, text.strip(), platform_style, creative_type, progress
                ))
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # 尝试使用新的 event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._process_async(image, text.strip(), platform_style, creative_type, progress)
                        )
                    finally:
                        loop.close()
                else:
                    raise

    def build_ui(self) -> gr.Blocks:
        """
        构建 Gradio UI

        Returns:
            Gradio Blocks 应用
        """
        with gr.Blocks(
            title="Re-Value Agent - 二手商品图文创意自动化",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown(
                """
                # Re-Value Agent
                ### 二手商品图文创意自动化视觉 Agent

                输入随手拍摄的商品原图 + 简述，自动完成商品特征提取、高质量抠图、
                场景重绘、文案生成及图文合成，输出可直接用于发布的电商级主图和营销文案。
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # 输入区域
                    image_input = gr.Image(
                        label="商品图片",
                        type="pil",
                        height=300,
                    )

                    text_input = gr.Textbox(
                        label="商品简述",
                        placeholder="例如：九成新办公椅，100元转让",
                        lines=2,
                    )

                    with gr.Row():
                        platform_select = gr.Dropdown(
                            label="平台风格",
                            choices=["闲鱼体", "小红书体"],
                            value="闲鱼体",
                            scale=1,
                        )

                        creative_select = gr.Dropdown(
                            label="创意装饰",
                            choices=["全部", "闪光", "星星", "心形", "徽章", "价格标签"],
                            value="全部",
                            scale=1,
                        )

                    submit_btn = gr.Button(
                        "开始处理",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown(
                        """
                        ### 示例图片
                        点击下方示例自动填充
                        """
                    )

                    gr.Examples(
                        examples=self.examples,
                        inputs=[image_input, text_input],
                        label=None,
                    )

                with gr.Column(scale=1):
                    # 输出区域
                    final_image_output = gr.Image(
                        label="生成的主图",
                        type="pil",
                        height=300,
                    )

                    title_output = gr.Textbox(
                        label="生成标题",
                        lines=1,
                    )

                    content_output = gr.Textbox(
                        label="生成文案",
                        lines=4,
                    )

                    status_output = gr.Textbox(
                        label="处理状态",
                        lines=2,
                        interactive=False,
                    )

            # 事件绑定
            submit_btn.click(
                fn=self.process_image,
                inputs=[image_input, text_input, platform_select, creative_select],
                outputs=[final_image_output, title_output, content_output, status_output],
            )

            # GPU 状态提示
            if not self.has_cuda:
                gr.Markdown(
                    """
                    > ⚠️ 当前环境无 GPU，图像生成将使用占位图。
                    > 如需真实生成，请配置 CUDA 环境或使用云端 API。
                    """,
                    visible=True,
                )

        return app

    def launch(self, server_name: str = "0.0.0.0", server_port: int = 7860):
        """启动应用"""
        app = self.build_ui()
        logger.info("Starting Gradio app on %s:%d", server_name, server_port)
        app.launch(server_name=server_name, server_port=server_port, share=False)


def main():
    """主入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = ReValueGradioApp()
    app.launch()


if __name__ == "__main__":
    main()

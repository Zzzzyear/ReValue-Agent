"""
ReValue-Agent 完整流程批量测试
=================================

对 data/input 目录下的所有样本图片运行完整流程。

## 运行命令

```bash
# 1. 激活环境
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# 2. 运行完整流程（处理所有样本）
python tests/run_full_pipeline.py

# 3. 或指定特定图片
python tests/run_full_pipeline.py --image data/input/xianyu_bag_01.jpg

# 4. 指定输出目录
python tests/run_full_pipeline.py --output data/output/batch_test
```

## 输入文件
- data/input/*.jpg - 商品图片
- data/input/*.json - 对应描述文件

## 输出文件
- *_final.png - 融合后的商品图
- *_decorated.png - 带装饰和文案的最终图
- *_copy.json - 生成的营销文案
- *_full_report.json - 完整日志报告
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import ReValueEngine
from core.context import Context, TaskStatus
from core.prompts import get_vlm_prompts, parse_json_from_response
from tools.api_clients import get_vlm_client, get_llm_client
from tools.cv_utils import MattingTool, ShadowRenderer, ImageFusion
from tools.sd_local import get_sd_generator
from tools.text_renderer import TextRenderer
from tools.decorations import DecorationRenderer

# 配置日志
log_file = Path("data/output/pipeline_log.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置"""
    import yaml
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class FullPipelineRunner:
    """完整流程运行器"""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.matting_tool = MattingTool(model_name="u2net")
        self.shadow_renderer = ShadowRenderer()
        self.fusion = ImageFusion()
        self.text_renderer = TextRenderer()
        self.decorations = DecorationRenderer()
        self.sd_generator = None  # 延迟加载

    def _get_sd_generator(self):
        if self.sd_generator is None:
            self.sd_generator = get_sd_generator(self.config)
        return self.sd_generator

    async def run_single(
        self,
        image_path: str,
        description: str,
        output_dir: str = "data/output",
        platform: str = "闲鱼体"
    ) -> dict:
        """运行单个图片的完整流程"""
        logger.info("=" * 60)
        logger.info(f"处理图片: {image_path}")
        logger.info(f"描述: {description}")
        logger.info(f"平台: {platform}")
        logger.info("=" * 60)

        start_time = time.time()
        result = {
            "image_path": image_path,
            "description": description,
            "platform": platform,
            "steps": {},
            "output_files": {},
            "status": "running",
            "error": None
        }

        try:
            # 加载图片
            img = Image.open(image_path)
            result["steps"]["image_loaded"] = {
                "size": img.size,
                "mode": img.mode
            }
            logger.info(f"[1/7] 图片加载: {img.size}")

            # Step 1: VLM 分析
            logger.info("[2/7] VLM 分析中...")
            vlm_result = await self._call_vlm(img, description)
            result["steps"]["vlm_analysis"] = vlm_result
            logger.info(f"  subject: {vlm_result.get('subject')}")
            logger.info(f"  selling_points: {vlm_result.get('selling_points', [])[:2]}...")

            # Step 2: 文案生成
            logger.info("[3/7] 文案生成中...")
            copy_result = await self._generate_copy(
                subject=vlm_result.get('subject', '商品'),
                selling_points=vlm_result.get('selling_points', []),
                original_text=description,
                platform=platform
            )
            result["steps"]["copy_generation"] = copy_result
            logger.info(f"  title: {copy_result.get('title', '')[:50]}")
            logger.info(f"  content: {copy_result.get('content', '')[:50]}...")

            # Step 3: 抠图
            logger.info("[4/7] 抠图中...")
            matted = self.matting_tool.remove_background(img)
            matted_path = Path(output_dir) / f"{Path(image_path).stem}_matted.png"
            matted.save(matted_path)
            result["steps"]["matting"] = {"path": str(matted_path)}
            result["output_files"]["matted"] = str(matted_path)
            logger.info(f"  已保存: {matted_path}")

            # Step 4: SD 背景生成
            logger.info("[5/7] SD 背景生成中...")
            bg_prompt = vlm_result.get('background_prompt', 'luxury boutique interior')
            bg = await self._generate_background(bg_prompt)
            bg_path = Path(output_dir) / f"{Path(image_path).stem}_background.png"
            bg.save(bg_path)
            result["steps"]["background_generation"] = {
                "prompt": bg_prompt,
                "path": str(bg_path)
            }
            result["output_files"]["background"] = str(bg_path)
            logger.info(f"  已保存: {bg_path}")

            # Step 5: 图像融合
            logger.info("[6/7] 图像融合中...")
            final = self.fusion.fuse(matted, bg, shadow_renderer=self.shadow_renderer)
            final_path = Path(output_dir) / f"{Path(image_path).stem}_final.png"
            final.save(final_path)
            result["steps"]["fusion"] = {"path": str(final_path)}
            result["output_files"]["final"] = str(final_path)
            logger.info(f"  已保存: {final_path}")

            # Step 6: 装饰渲染
            logger.info("[7/7] 装饰渲染中...")
            decorated = self._apply_decorations(final)
            decorated_path = Path(output_dir) / f"{Path(image_path).stem}_decorated.png"
            decorated.save(decorated_path)
            result["steps"]["decorations"] = {"path": str(decorated_path)}
            result["output_files"]["decorated"] = str(decorated_path)
            logger.info(f"  已保存: {decorated_path}")

            # Step 7: 文字渲染
            title = copy_result.get('title', description)
            final_with_text = self.text_renderer.render(
                image=decorated,
                text=title,
                position="bottom-center",
                max_width_ratio=0.85,
                font_size=48
            )
            final_with_text_path = Path(output_dir) / f"{Path(image_path).stem}_final_with_text.png"
            final_with_text.save(final_with_text_path)
            result["output_files"]["final_with_text"] = str(final_with_text_path)
            logger.info(f"  已保存: {final_with_text_path}")

            # 保存文案
            copy_path = Path(output_dir) / f"{Path(image_path).stem}_copy.json"
            with open(copy_path, 'w', encoding='utf-8') as f:
                json.dump(copy_result, f, ensure_ascii=False, indent=2)
            result["output_files"]["copy"] = str(copy_path)

            # 保存完整报告
            elapsed = time.time() - start_time
            result["elapsed_time"] = elapsed
            result["status"] = "completed"

            report = {
                "task_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": image_path,
                    "description": description,
                    "platform": platform,
                    "elapsed_time": f"{elapsed:.1f}秒",
                    "status": "completed"
                },
                "vlm_result": vlm_result,
                "copy_result": copy_result,
                "output_files": result["output_files"]
            }
            report_path = Path(output_dir) / f"{Path(image_path).stem}_full_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            result["output_files"]["report"] = str(report_path)

            logger.info("=" * 60)
            logger.info(f"✅ 处理完成! 耗时 {elapsed:.1f}秒")
            logger.info("=" * 60)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"❌ 处理失败: {e}", exc_info=True)

        return result

    async def _call_vlm(self, image: Image.Image, text: str) -> dict:
        """调用 VLM"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        prompts = get_vlm_prompts(text)
        client = get_vlm_client("qwen3.5-plus", self.config)

        response = await client.chat_with_image(
            image_base64=image_base64,
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
        )

        return parse_json_from_response(response)

    async def _generate_copy(
        self,
        subject: str,
        selling_points: list,
        original_text: str,
        platform: str
    ) -> dict:
        """生成文案"""
        from core.node_text import NodeText

        node_text = NodeText(self.config)
        ctx = Context(
            subject=subject,
            selling_points=selling_points,
            original_text=original_text,
            platform_style=platform,
        )
        ctx = await node_text.process(ctx)
        return ctx.final_copy or {"title": subject, "content": original_text}

    async def _generate_background(self, prompt: str) -> Image.Image:
        """生成背景"""
        sd = self._get_sd_generator()
        return await sd.generate(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
        )

    def _apply_decorations(self, image: Image.Image) -> Image.Image:
        """应用装饰"""
        elements = [
            {"type": "sparkle", "position": "top-right", "count": 3, "size": "medium"},
            {"type": "heart", "position": "top-right", "count": 2, "size": "medium"},
            {"type": "badge", "position": "bottom-left", "text": "SALE", "style": "rounded_rect"},
        ]
        return self.decorations.render_elements(image, elements)


def load_input_samples(input_dir: str = "data/input") -> list:
    """加载输入样本"""
    samples = []
    input_path = Path(input_dir)

    for json_file in input_path.glob("*.json"):
        img_file = json_file.with_suffix(".jpg")
        if img_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            samples.append({
                "image": str(img_file),
                "description": data.get("description", ""),
                "platform": data.get("platform", "闲鱼"),
            })

    return samples


async def run_all_samples(input_dir: str = "data/input", output_dir: str = "data/output"):
    """运行所有样本"""
    logger.info("=" * 60)
    logger.info("ReValue-Agent 完整流程批量测试")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # 加载配置和样本
    config = load_config()
    samples = load_input_samples(input_dir)
    logger.info(f"找到 {len(samples)} 个样本")
    logger.info("")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 运行每个样本
    runner = FullPipelineRunner(config)
    results = []

    for i, sample in enumerate(samples, 1):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# 处理进度: {i}/{len(samples)}")
        logger.info(f"{'#' * 60}")

        result = await runner.run_single(
            image_path=sample["image"],
            description=sample["description"],
            output_dir=output_dir,
            platform=sample.get("platform", "闲鱼体")
        )
        results.append(result)

    # 汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("批量处理汇总")
    logger.info("=" * 60)

    success = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"总计: {len(results)} 个样本")
    logger.info(f"成功: {success} 个")
    logger.info(f"失败: {failed} 个")
    logger.info(f"日志文件: {log_file}")

    for r in results:
        status_icon = "✅" if r["status"] == "completed" else "❌"
        logger.info(f"{status_icon} {Path(r['image_path']).name}: {r['status']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReValue-Agent 完整流程测试")
    parser.add_argument("--image", "-i", help="指定单个图片路径")
    parser.add_argument("--input", default="data/input", help="输入目录")
    parser.add_argument("--output", "-o", default="data/output", help="输出目录")

    args = parser.parse_args()

    if args.image:
        # 运行单个图片
        import json
        json_file = Path(args.image).with_suffix(".json")
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            description = data.get("description", "")
        else:
            description = "二手商品"

        runner = FullPipelineRunner()
        asyncio.run(runner.run_single(
            image_path=args.image,
            description=description,
            output_dir=args.output
        ))
    else:
        # 运行所有样本
        asyncio.run(run_all_samples(args.input, args.output))

"""
ReValue-Agent Prompt 调试脚本
=============================

用于调试和优化 VLM/Copy/SB 的 prompt 效果。

## 运行命令

```bash
cd D:/Code/AI_Playground/projects/ReValue-Agent
venv\Scripts\activate

# VLM Prompt 调试（分析图片）
python tests/debug_prompts.py --mode vlm -i data/input/xianyu_bag_01.jpg -t "MiniKelly二手包"

# Copy Prompt 调试（生成文案）
python tests/debug_prompts.py --mode copy -s "运动鞋" -p "九成新|实战利器|李宁" -o "李宁球鞋便宜卖"

# SD Prompt 调试（生成背景）
python tests/debug_prompts.py --mode sd -p "luxury boutique interior"

# 批量对比测试（多个样本）
python tests/debug_prompts.py --mode batch -i data/input

# 交互模式
python tests/debug_prompts.py --mode interactive
```

## 功能说明

1. **VLM Mode**: 测试 VLM 对图片的分析结果
   - subject 是否准确
   - selling_points 是否吸引人
   - background_prompt 是否合理
   - creative_elements 是否合适

2. **Copy Mode**: 测试文案生成效果
   - 标题是否吸睛
   - 正文是否自然
   - Emoji 是否适量

3. **SD Mode**: 测试背景生成效果
   - 直接用 prompt 生成背景图

4. **Batch Mode**: 批量测试多个样本
   - 对比不同图片的 prompt 效果

5. **Interactive Mode**: 交互式调试
   - 实时修改 prompt 参数
   - 立即看到输出效果
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
from pathlib import Path
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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


# ============================================================================
# VLM Prompt 调试
# ============================================================================

async def debug_vlm_prompt(
    image_path: str = None,
    text: str = "",
    custom_system: str = None,
    custom_user: str = None,
    output_dir: str = "data/output/debug",
) -> dict:
    """
    调试 VLM Prompt

    Args:
        image_path: 图片路径
        text: 原始描述
        custom_system: 自定义 system prompt
        custom_user: 自定义 user prompt
        output_dir: 输出目录
    """
    from tools.api_clients import get_vlm_client
    from core.prompts import get_vlm_prompts, parse_json_from_response, VLM_SYSTEM_PROMPT, VLM_USER_PROMPT_TEMPLATE

    print("\n" + "=" * 60)
    print("VLM Prompt 调试")
    print("=" * 60)

    # 加载图片
    if image_path:
        img = Image.open(image_path)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        print(f"图片: {image_path} ({img.size})")
    else:
        image_base64 = None
        print("无图片")

    print(f"描述: {text}")
    print("-" * 60)

    # 使用自定义或默认 prompt
    if custom_system:
        system_prompt = custom_system
    else:
        system_prompt = VLM_SYSTEM_PROMPT

    if custom_user:
        user_prompt = custom_user
    else:
        user_prompt = VLM_USER_PROMPT_TEMPLATE.format(original_text=text)

    print("【System Prompt】")
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    print("-" * 60)
    print("【User Prompt】")
    print(user_prompt)
    print("-" * 60)

    # 调用 VLM
    config = load_config()
    client = get_vlm_client("qwen3.5-plus", config)

    try:
        if image_base64:
            response = await client.chat_with_image(
                image_base64=image_base64,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            # 无图片时的纯文本调用
            response = await client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        print("【原始响应】")
        print(response.encode('utf-8', errors='replace').decode('utf-8'))
        print("-" * 60)

        # 解析 JSON
        result = parse_json_from_response(response)
        print("【解析结果】")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        debug_result = {
            "mode": "vlm",
            "image_path": image_path,
            "text": text,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": response,
            "parsed_result": result,
        }
        result_file = output_path / "vlm_debug_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(debug_result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {result_file}")

        return result

    except Exception as e:
        print(f"❌ VLM 调用失败: {e}")
        return None


# ============================================================================
# Copy Prompt 调试
# ============================================================================

async def debug_copy_prompt(
    subject: str = "商品",
    selling_points: str = "二手好物|超低价|实用",
    original_text: str = "",
    platform: str = "闲鱼体",
    custom_system: str = None,
    custom_user: str = None,
    output_dir: str = "data/output/debug",
) -> dict:
    """
    调试 Copy Prompt

    Args:
        subject: 商品名称
        selling_points: 卖点（用 | 分隔）
        original_text: 原始描述
        platform: 平台风格
        custom_system: 自定义 system prompt
        custom_user: 自定义 user prompt
        output_dir: 输出目录
    """
    from tools.api_clients import get_llm_client
    from core.prompts import get_copy_prompts, COPY_SYSTEM_PROMPT_TEMPLATE, COPY_USER_PROMPT_TEMPLATE

    print("\n" + "=" * 60)
    print("Copy Prompt 调试")
    print("=" * 60)

    # 解析卖点
    sp_list = [s.strip() for s in selling_points.split('|') if s.strip()]
    if not sp_list:
        sp_list = ["二手好物", "超低价", "实用性强"]

    print(f"商品: {subject}")
    print(f"卖点: {sp_list}")
    print(f"原始描述: {original_text}")
    print(f"平台: {platform}")
    print("-" * 60)

    # 使用自定义或默认 prompt
    if custom_system:
        system_prompt = custom_system
    else:
        system_prompt = COPY_SYSTEM_PROMPT_TEMPLATE

    if custom_user:
        user_prompt = custom_user
    else:
        # 获取平台配置
        platform_configs = {
            "闲鱼体": {"emoji_enabled": True, "max_title_length": 30, "max_content_length": 200},
            "小红书体": {"emoji_enabled": True, "hashtag_enabled": True, "max_title_length": 20, "max_content_length": 500},
        }
        platform_config = platform_configs.get(platform, platform_configs["闲鱼体"])
        platform_style_str = f"平台：{platform}\n特点：Emoji={platform_config['emoji_enabled']}, 标题≤{platform_config['max_title_length']}字, 正文≤{platform_config['max_content_length']}字"

        user_prompt = COPY_USER_PROMPT_TEMPLATE.format(
            subject=subject,
            selling_points=", ".join(sp_list),
            original_text=original_text,
        )
        system_prompt = COPY_SYSTEM_PROMPT_TEMPLATE.format(platform_style=platform_style_str)

    print("【System Prompt】")
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    print("-" * 60)
    print("【User Prompt】")
    print(user_prompt)
    print("-" * 60)

    # 调用 LLM
    config = load_config()
    client = get_llm_client("qwen-plus", config)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await client.chat(messages=messages)

        print("【原始响应】")
        logger.info(response)
        print("-" * 60)

        # 解析 JSON
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"title": response.split('\n')[0], "content": response}

        print("【解析结果】")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        debug_result = {
            "mode": "copy",
            "subject": subject,
            "selling_points": sp_list,
            "original_text": original_text,
            "platform": platform,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": response,
            "parsed_result": result,
        }
        result_file = output_path / "copy_debug_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(debug_result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {result_file}")

        return result

    except Exception as e:
        logger.error(f"Copy 调用失败: {e}")
        return None


# ============================================================================
# SD Prompt 调试
# ============================================================================

async def debug_sd_prompt(
    prompt: str = "luxury boutique interior",
    output_dir: str = "data/output/debug",
    width: int = 1024,
    height: int = 1024,
) -> Image.Image:
    """
    调试 SD Prompt

    Args:
        prompt: 背景描述
        output_dir: 输出目录
        width: 宽度
        height: 高度
    """
    from tools.sd_local import get_sd_generator

    print("\n" + "=" * 60)
    print("SD Prompt 调试")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"尺寸: {width}x{height}")
    print("-" * 60)

    try:
        config = load_config()
        sd = get_sd_generator(config)

        print("正在生成...")
        result = await sd.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.5,
        )

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 添加后缀
        from core.prompts import get_sd_prompt_suffix
        suffix = get_sd_prompt_suffix()

        # 生成文件名
        safe_prompt = prompt.replace(' ', '_')[:30]
        result_file = output_path / f"sd_{safe_prompt}_{width}x{height}.png"
        result.save(result_file)

        print(f"✅ 已保存: {result_file}")

        return result

    except Exception as e:
        print(f"❌ SD 生成失败: {e}")
        return None


# ============================================================================
# 批量调试
# ============================================================================

async def debug_batch(
    input_dir: str = "data/input",
    output_dir: str = "data/output/debug/batch",
):
    """批量测试多个样本"""
    from pathlib import Path

    print("\n" + "=" * 60)
    print("批量 Prompt 调试")
    print("=" * 60)

    samples = []
    input_path = Path(input_dir)

    # 加载样本
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

    print(f"找到 {len(samples)} 个样本")
    print("-" * 60)

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] 处理: {Path(sample['image']).name}")

        # VLM 分析
        vlm_result = await debug_vlm_prompt(
            image_path=sample["image"],
            text=sample["description"],
            output_dir=output_dir,
        )

        if vlm_result:
            # Copy 生成
            copy_result = await debug_copy_prompt(
                subject=vlm_result.get("subject", "商品"),
                selling_points="|".join(vlm_result.get("selling_points", [])),
                original_text=sample["description"],
                platform=sample.get("platform", "闲鱼体"),
                output_dir=output_dir,
            )

            results.append({
                "image": sample["image"],
                "vlm_result": vlm_result,
                "copy_result": copy_result,
            })

    # 保存汇总报告
    report_file = Path(output_dir) / "batch_debug_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"批量调试完成，结果已保存: {report_file}")

    return results


# ============================================================================
# 交互式调试
# ============================================================================

async def debug_interactive():
    """交互式调试模式"""
    import asyncio

    print("\n" + "=" * 60)
    print("交互式 Prompt 调试")
    print("=" * 60)
    print("命令:")
    print("  vlm <图片路径> <描述>  - 调试 VLM")
    print("  copy <商品> <卖点1|卖点2> <原始描述> - 调试 Copy")
    print("  sd <prompt>            - 调试 SD")
    print("  prompt <vlm|copy>      - 查看当前 prompt")
    print("  set <key> <value>      - 修改参数")
    print("  quit                   - 退出")
    print("=" * 60)

    # 存储当前参数
    params = {
        "image": None,
        "text": "",
        "subject": "商品",
        "selling_points": "二手好物|超低价",
        "platform": "闲鱼体",
    }

    while True:
        try:
            cmd = input("\n> ").strip()

            if not cmd:
                continue

            parts = cmd.split(maxsplit=2)
            action = parts[0].lower()

            if action == "quit" or action == "q":
                print("退出...")
                break

            elif action == "vlm":
                if len(parts) >= 2:
                    params["image"] = parts[1]
                if len(parts) >= 3:
                    params["text"] = parts[2]
                await debug_vlm_prompt(params["image"], params["text"])

            elif action == "copy":
                if len(parts) >= 2:
                    params["subject"] = parts[1]
                if len(parts) >= 3:
                    params["selling_points"] = parts[2]
                await debug_copy_prompt(params["subject"], params["selling_points"], params["text"], params["platform"])

            elif action == "sd":
                prompt = parts[1] if len(parts) >= 2 else "luxury boutique interior"
                await debug_sd_prompt(prompt)

            elif action == "prompt":
                if len(parts) >= 2 and parts[1] == "vlm":
                    from core.prompts import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT_TEMPLATE
                    print("\n【VLM System Prompt】")
                    print(VLM_SYSTEM_PROMPT)
                    print("\n【VLM User Prompt Template】")
                    print(VLM_USER_PROMPT_TEMPLATE)
                elif len(parts) >= 2 and parts[1] == "copy":
                    from core.prompts import COPY_SYSTEM_PROMPT_TEMPLATE, COPY_USER_PROMPT_TEMPLATE
                    print("\n【Copy System Prompt Template】")
                    print(COPY_SYSTEM_PROMPT_TEMPLATE)
                    print("\n【Copy User Prompt Template】")
                    print(COPY_USER_PROMPT_TEMPLATE)
                else:
                    print("用法: prompt <vlm|copy>")

            elif action == "set":
                if len(parts) >= 3:
                    key = parts[1]
                    value = parts[2]
                    params[key] = value
                    print(f"已设置 {key} = {value}")
                else:
                    print("用法: set <key> <value>")

            elif action == "help":
                print("命令:")
                print("  vlm <图片路径> <描述>  - 调试 VLM")
                print("  copy <商品> <卖点1|卖点2> <原始描述> - 调试 Copy")
                print("  sd <prompt>            - 调试 SD")
                print("  prompt <vlm|copy>      - 查看当前 prompt")
                print("  set <key> <value>      - 修改参数")
                print("  quit                   - 退出")

            else:
                print(f"未知命令: {action}，输入 help 查看帮助")

        except KeyboardInterrupt:
            print("\n退出...")
            break
        except Exception as e:
            print(f"错误: {e}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ReValue-Agent Prompt 调试工具")
    parser.add_argument("--mode", "-m", choices=["vlm", "copy", "sd", "batch", "interactive"],
                        default="interactive", help="调试模式")
    parser.add_argument("--image", "-i", help="图片路径 (VLM模式)")
    parser.add_argument("--text", "-t", default="", help="原始描述 / 商品信息")
    parser.add_argument("--subject", "-s", default="商品", help="商品名称 (Copy模式)")
    parser.add_argument("--selling-points", "-p", default="二手好物|超低价|实用", help="卖点，用|分隔")
    parser.add_argument("--original-text", "-o", default="", help="原始描述")
    parser.add_argument("--platform", "-pt", default="闲鱼体", help="平台风格")
    parser.add_argument("--prompt", "-pm", default="", help="SD prompt / 自定义 prompt")
    parser.add_argument("--system-prompt", help="自定义 system prompt")
    parser.add_argument("--user-prompt", help="自定义 user prompt")
    parser.add_argument("--output", "-out", default="data/output/debug", help="输出目录")
    parser.add_argument("--width", "-w", type=int, default=1024, help="SD 图像宽度")
    parser.add_argument("--height", "-ht", type=int, default=1024, help="SD 图像高度")
    parser.add_argument("--input-dir", help="批量测试输入目录")

    args = parser.parse_args()

    if args.mode == "vlm":
        asyncio.run(debug_vlm_prompt(
            image_path=args.image,
            text=args.text or args.original_text,
            custom_system=args.system_prompt,
            custom_user=args.user_prompt,
            output_dir=args.output,
        ))

    elif args.mode == "copy":
        asyncio.run(debug_copy_prompt(
            subject=args.subject,
            selling_points=args.selling_points,
            original_text=args.original_text or args.text,
            platform=args.platform,
            custom_system=args.system_prompt,
            custom_user=args.user_prompt,
            output_dir=args.output,
        ))

    elif args.mode == "sd":
        prompt = args.prompt or "luxury boutique interior"
        asyncio.run(debug_sd_prompt(
            prompt=prompt,
            output_dir=args.output,
            width=args.width,
            height=args.height,
        ))

    elif args.mode == "batch":
        asyncio.run(debug_batch(
            input_dir=args.input_dir or "data/input",
            output_dir=args.output,
        ))

    elif args.mode == "interactive":
        asyncio.run(debug_interactive())


if __name__ == "__main__":
    main()

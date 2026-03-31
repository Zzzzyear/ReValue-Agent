"""
Re-Value Agent - Prompt 集中管理
存放 Node 1 (VLM Router) 和 Node 2B (文案生成) 的所有系统模板
"""

from typing import Dict


# =============================================================================
# Node 1: VLM 联合感知与路由分发 Prompt
# =============================================================================

VLM_SYSTEM_PROMPT = """你是一个专业的电商图像分析助手。你的任务是从用户提供的商品图片和描述中提取关键信息，生成结构化的JSON输出。

## 输出要求
必须严格输出JSON格式，不要包含任何其他文字。JSON格式如下：
{
    "subject": "商品核心品类（如：办公椅、书架、茶几）",
    "selling_points": ["卖点1", "卖点2", "卖点3", "卖点4"],
    "background_prompt": "一段英文商业摄影提示词，用于生成专业背景图",
    "reference_box": [x_min, y_min, x_max, y_max],
    "creative_elements": [
        {
            "type": "sparkle",
            "position": "top-right",
            "count": 3,
            "size": "small"
        },
        {
            "type": "price_badge",
            "position": "bottom-left",
            "text": "2.8w",
            "style": "rounded_rect"
        }
    ],
    "text_placement": {
        "region": "bottom_center",
        "max_width_ratio": 0.8
    }
}

## 输出规范
- subject: 简洁明确的商品品类名称，2-5个字
- selling_points: 3-4个核心卖点，结合成色和价格提炼，要具体有吸引力
- background_prompt: 英文，描述商业摄影风格背景，包含光线、氛围、风格等要素，以"8k, professional product photography"结尾
- reference_box: 如果商品主体明确，给出其在图片中的相对边界框坐标 [左, 上, 右, 下]，范围0-1。如果不确定，设为 null

## creative_elements 装饰元素说明
- type: 装饰类型，可选值：
  - sparkle: 闪光/星星
  - star: 星形
  - heart: 心形
  - badge: 徽章
  - price_tag: 价格标签
  - ribbon: 丝带
  - circle_decoration: 圆形装饰
- position: 位置，可选值：top-left, top-right, bottom-left, bottom-right, top-center, bottom-center, center
- count: 数量，1-5，对 sparkle/star/heart 有效
- size: 大小，可选值：small, medium, large
- text: 标签文字，仅对 badge/price_tag 有效
- style: 样式，仅对 badge 有效，可选值：rounded_rect, circle, star

## text_placement 文字位置说明
- region: 文字放置区域，可选值：top_center, bottom_center, overlay
- max_width_ratio: 文字最大宽度占图像宽度的比例，0.5-0.9

## 示例输出
输入：九成新办公椅，100元
输出：{"subject": "办公椅", "selling_points": ["九成新", "100元超低价", "黑色经典款", "可调节高度"], "background_prompt": "minimalist office interior, soft natural lighting from window, clean white wall, modern workspace ambiance, 8k, professional product photography", "reference_box": [0.3, 0.2, 0.7, 0.8], "creative_elements": [{"type": "sparkle", "position": "top-right", "count": 3, "size": "small"}, {"type": "price_tag", "position": "bottom-left", "text": "100元", "style": "rounded_rect"}], "text_placement": {"region": "bottom_center", "max_width_ratio": 0.8}}

注意：只输出JSON，不要有任何其他内容。"""


VLM_USER_PROMPT_TEMPLATE = """## 商品图片
请分析以下商品图片。

## 用户描述
{original_text}

## 请提取商品信息并输出JSON"""


# Fallback JSON (兜底策略)
FALLBACK_JSON = {
    "subject": "商品",
    "selling_points": ["二手好物", "超低价", "实用性强", "性价比高"],
    "background_prompt": "minimalist white background, soft studio lighting, clean and bright atmosphere, 8k, professional product photography",
    "reference_box": None,
    "creative_elements": [
        {"type": "sparkle", "position": "top-right", "count": 3, "size": "small"},
    ],
    "text_placement": {"region": "bottom_center", "max_width_ratio": 0.8},
}


# =============================================================================
# Node 2B: 文案生成 Prompt
# =============================================================================

COPY_SYSTEM_PROMPT_TEMPLATE = """你是一个专业的二手商品文案撰写专家，擅长写闲鱼/小红书风格的销售文案。

## 你的角色
- 熟悉二手交易平台的文化和用户心理
- 能够挖掘商品卖点，撰写有吸引力的文案
- 语气亲切、自然、有生活气息

## 文案结构
1. **吸睛标题**：简短有力，包含商品核心优势和价格
2. **正文描述**：3-5句话，介绍商品来源、成色、使用感受
3. **强调卖点**：突出性价比和核心优势

## 风格要求
- 使用适当的 Emoji 增添活力
- 避免过于商业化的语言
- 可以加入一些小故事或使用场景描述
- 价格要醒目

## 平台风格选择
{platform_style}

## 输出格式
必须输出JSON格式：
{{"title": "标题", "content": "正文内容"}}

注意：只输出JSON，不要有任何其他内容。"""


COPY_USER_PROMPT_TEMPLATE = """## 商品信息
- 商品名称：{subject}
- 核心卖点：{selling_points}
- 原始描述：{original_text}

## 重要提示
- 如果商品名称是"商品"等通用词，请务必根据【原始描述】和【核心卖点】来推断真实商品
- 原始描述中往往包含商品的真实名称、品牌、型号、成色、价格等信息
- 例如：原始描述"Nike足球鞋九成新"说明这是Nike足球鞋，九成新

## 请生成闲鱼风格的销售文案
1. 根据商品信息发挥创意，生成有吸引力的标题
2. 标题要突出品牌、商品名称、成色、价格等关键信息
3. 正文要自然亲切，可以描述使用场景或转让原因
4. 适量使用 Emoji"""


# =============================================================================
# Prompt 辅助函数
# =============================================================================

def get_vlm_prompts(original_text: str) -> Dict[str, str]:
    """
    获取 VLM Router 的完整 Prompt

    Args:
        original_text: 用户输入的原始描述

    Returns:
        Dict with 'system' and 'user' prompts
    """
    return {
        "system": VLM_SYSTEM_PROMPT,
        "user": VLM_USER_PROMPT_TEMPLATE.format(original_text=original_text),
    }


def get_copy_prompts(
    subject: str,
    selling_points: list,
    original_text: str,
    platform_style: str = "闲鱼体",
) -> Dict[str, str]:
    """
    获取文案生成的完整 Prompt

    Args:
        subject: 商品品类名称
        selling_points: 卖点列表
        original_text: 原始描述
        platform_style: 平台风格 (闲鱼体/小红书体)

    Returns:
        Dict with 'system' and 'user' prompts
    """
    # 获取平台风格配置
    platform_configs = {
        "闲鱼体": {
            "emoji_enabled": True,
            "max_title_length": 30,
            "max_content_length": 200,
        },
        "小红书体": {
            "emoji_enabled": True,
            "hashtag_enabled": True,
            "max_title_length": 20,
            "max_content_length": 500,
        },
    }

    platform_config = platform_configs.get(platform_style, platform_configs["闲鱼体"])
    platform_style_str = f"平台：闲鱼\n特点：Emoji={platform_config['emoji_enabled']}, 标题≤{platform_config['max_title_length']}字, 正文≤{platform_config['max_content_length']}字"

    return {
        "system": COPY_SYSTEM_PROMPT_TEMPLATE.format(platform_style=platform_style_str),
        "user": COPY_USER_PROMPT_TEMPLATE.format(
            subject=subject,
            selling_points=", ".join(selling_points),
            original_text=original_text,
        ),
    }


def get_sd_prompt_suffix() -> str:
    """
    获取追加到 SD background_prompt 的高画质后缀

    Returns:
        高画质后缀字符串
    """
    return ", highly detailed, 8k resolution, ultra sharp focus, soft shadows, professional photography"


def parse_json_from_response(response: str) -> Dict:
    """
    从 VLM 响应中解析 JSON

    使用多种策略:
    1. 直接解析完整JSON
    2. 提取 ```json ... ``` 块
    3. 提取 {} 内的内容
    4. 使用正则表达式提取各字段

    Args:
        response: VLM 原始响应

    Returns:
        解析后的 JSON dict
    """
    import json
    import re

    # 策略1: 直接尝试解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 策略2: 提取 ```json ... ``` 块
    try:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            return json.loads(json_match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 策略3: 提取 {} 内的内容
    try:
        # 尝试找最外层的 {}
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # 策略4: 使用正则表达式提取各字段（基础版本）
    try:
        result = {}

        # 提取 subject
        subject_match = re.search(r'"subject"\s*:\s*"([^"]*)"', response)
        if subject_match:
            result["subject"] = subject_match.group(1)

        # 提取 selling_points
        sp_match = re.search(r'"selling_points"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if sp_match:
            points = re.findall(r'"([^"]*)"', sp_match.group(1))
            result["selling_points"] = points if points else []

        # 提取 background_prompt
        bg_match = re.search(r'"background_prompt"\s*:\s*"([^"]*)"', response)
        if bg_match:
            result["background_prompt"] = bg_match.group(1)

        # 提取 reference_box
        box_match = re.search(r'"reference_box"\s*:\s*\[([^\]]*)\]', response)
        if box_match:
            box_str = box_match.group(1)
            box_values = re.findall(r'[\d.]+', box_str)
            if len(box_values) == 4:
                result["reference_box"] = [float(v) for v in box_values]
            else:
                result["reference_box"] = None
        else:
            result["reference_box"] = None

        # 提取 creative_elements (简化处理)
        ce_match = re.search(r'"creative_elements"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if ce_match:
            result["creative_elements"] = [{"type": "sparkle", "position": "top-right", "count": 3, "size": "small"}]
        else:
            result["creative_elements"] = FALLBACK_JSON.get("creative_elements", [])

        # 提取 text_placement
        tp_match = re.search(r'"text_placement"\s*:\s*\{([^}]*)\}', response, re.DOTALL)
        if tp_match:
            result["text_placement"] = {"region": "bottom_center", "max_width_ratio": 0.8}
        else:
            result["text_placement"] = FALLBACK_JSON.get("text_placement", {"region": "bottom_center", "max_width_ratio": 0.8})

        if result.get("subject"):
            return result

    except Exception:
        pass

    # 所有策略都失败，返回 FALLBACK_JSON
    return FALLBACK_JSON

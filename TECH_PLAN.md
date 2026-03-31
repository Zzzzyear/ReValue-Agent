# ReValue-Agent 技术决策记录

> 本文档记录项目的关键技术决策和备选方案，供后续参考。

---

## 技术决策点

### 1. SD 模型选择

| 选项 | 显存需求 | 出图速度 | 质量 | 推荐度 |
|------|----------|----------|------|--------|
| SD 1.5 + Lora | 4-6GB | 中等 | 较好 | ⭐⭐⭐⭐ |
| SDXL-Turbo | 4-6GB | 快速（1-4步） | 一般 | ⭐⭐⭐ |
| SDXL原生 | 12GB+ | 慢 | 最佳 | ⭐ |
| 云端API(备选) | 0 | 中等 | 较好 | ⭐⭐⭐ |

**已选**：SD 1.5 + 电商风格Lora，4060流畅运行

### 2. 任务队列策略

| 选项 | 复杂度 | 适用场景 | 推荐度 |
|------|--------|----------|--------|
| asyncio + run_in_executor | 低 | 5用户Demo | ⭐⭐⭐⭐ |
| RQ (Redis Queue) | 中 | 扩展到10+用户 | ⭐⭐⭐ |
| Celery | 高 | 正式生产服务 | ⭐⭐ |

**已选**：asyncio + run_in_executor 简化

### 3. 抠图工具

| 工具 | 精度 | 速度 | 显存 | 推荐度 |
|------|------|------|------|--------|
| rembg (u2net) | 中等 | 快 | 低 | ⭐⭐⭐⭐ |
| RMBG-1.4 (BRIA) | 高 | 中等 | 中 | ⭐⭐⭐⭐⭐ |
| SAM (Meta) | 高 | 慢 | 高 | ⭐⭐⭐ |

**已选**：rembg 为主，RMBG-1.4 为备选

---

## 已验证项 ✅

- [x] VLM JSON解析链路（qwen3.5-plus）
- [x] LLM 文案生成链路（qwen-plus）
- [x] rembg 抠图功能
- [x] SD 1.5 本地推理
- [x] 图像融合（阴影+叠加）
- [x] Node 2C 创意合成（装饰+文字渲染）
- [x] 模块化测试脚本（test_01_vlm, test_02_copy, test_03_vision, test_04_creative）

---

## 优化任务清单

### 🔴 P0 - 必须修复（阻塞性问题）

#### 1. 文字方框问题（字体不支持 Emoji）
**问题**：`config.yaml` 中配置的 `C:/Windows/Fonts/msyh.ttc`（微软雅黑）不支持 VLM/LLM 生成的 Emoji 字符，导致渲染出方框。

**修改文件**：`tools/text_renderer.py`

**解决方案**：
- 在 `render()` 方法中，使用正则表达式过滤掉非支持字符
- 或引入支持 Emoji 的复合字体（如 `Noto Color Emoji` 配合微软雅黑）

```python
# 示例：过滤 emoji
import re
def filter_unsupported_chars(text):
    # 移除 emoji 和无法渲染的字符
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)
```

---

#### 2. 文字重叠问题
**问题**：`tools/text_renderer.py` 中 `_wrap_text` 方法的换行与行高计算有 Bug，导致多行文字重叠。

**修改文件**：`tools/text_renderer.py`

**解决方案**：
- 检查 `font.getbbox()` 或 `font.getsize()` 的高度累加逻辑
- 确保每行之间有足够的 padding（建议 line_spacing = font_size * 1.2）
- 计算文本块总高度时正确累加

---

### 🟠 P1 - 高优先级（效果显著）

#### 3. 场景-品类硬路由（SCENE_TEMPLATES）
**问题**：VLM 生成的 background_prompt 太开放，导致足球鞋背景出现厨房、包包背景又生成包等不匹配问题。

**修改文件**：`core/node_vision.py`

**解决方案**：建立场景字典，根据 subject 硬路由

```python
SCENE_TEMPLATES = {
    "鞋": "placed on clean street surface, urban vibe, shallow depth of field",
    "包包": "displayed on marble pedestal, luxury boutique lighting, soft shadows",
    "手机": "on modern minimal desk, cyberpunk neon lights in background",
    "耳机": "resting on wooden desk, warm ambient lighting, minimalist setup",
    "键盘": "on gaming desk setup, RGB lighting ambiance",
    "手表": "displayed on velvet jewelry box, studio lighting, black background",
    "衣服": "arranged on clean white surface, fashion photography lighting",
    "玩具": "placed on kid's bedroom floor, warm natural lighting",
}
```

**逻辑**：如果 VLM 触发了 Fallback 或生成的 prompt 太短（<50字符），直接用 `SCENE_TEMPLATES.get(subject)` 兜底。

---

#### 4. VLM Prompt 优化（场景匹配 + 透视承托）
**问题**：背景图与原图视角不匹配（俯拍 vs 平视），商品悬空。

**修改文件**：`core/prompts.py`

**解决方案**：在 VLM_SYSTEM_PROMPT 中增加约束

```python
# 在 background_prompt 要求中增加：
"""
## 背景生成要求（必须严格遵守）
1. 相机视角：必须与原图商品视角一致（top-down view 俯拍 / eye-level shot 平视）
2. 物理承托面：商品必须有合理的"落脚点"
   - 鞋子：绿茵草地/街头地面/运动地板
   - 包包：云石台面/展示架/床
   - 电子产品：桌面/地面/专用支架
3. 严禁：悬空背景、与商品品类无关的场景（如厨房、浴室）

示例：
- 足球鞋 → "top-down view, placed on green grass field, stadium lighting"
- 办公椅 → "eye-level shot, standing on office carpet, natural window light"
- 手提包 → "displayed on white marble pedestal, luxury boutique lighting, soft shadows"
"""
```

---

### 🟡 P2 - 中优先级（体验提升）

#### 5. 光影一致性优化
**问题**：阴影方向固定向下，与 SD 生成的背景光照不匹配。

**修改文件**：
- `tools/cv_utils.py` (ShadowRenderer)
- `core/prompts.py` (VLM 输出光源方向)

**解决方案**：
1. VLM Prompt 要求输出 `light_direction` 字段（如 `top-right`, `left`, `bottom`）
2. ShadowRenderer 根据 `light_direction` 计算阴影偏移角度
3. SD prompt 中强制加入 `studio lighting, soft and consistent shadow`

---

#### 6. PNG 素材库 + 视觉效果升级
**问题**：PIL 代码手绘几何图形缺乏质感，装饰元素不显眼。

**修改文件**：
- `tools/decorations.py`
- 新建 `data/assets/` 目录

**解决方案**：

```
data/assets/
├── badges/
│   ├── sale_red.png        # 急售徽章
│   ├── new_pink.png        # 全新徽章
│   └── hot_orange.png      # 热卖徽章
├── sparkles/
│   ├── sparkle_gold.png    # 金色闪光
│   └── star_yellow.png     # 黄色星星
├── price_tags/
│   ├── tag_retro.png       # 复古价格牌
│   └── tag_modern.png      # 现代价格牌
└── emoji/
    ├── emoji_fire.png      # 🔥
    ├── emoji_heart.png     # ❤️
    └── emoji_money.png     # 💰
```

**代码逻辑**：
```python
def render_elements(self, image, elements):
    for elem in elements:
        if elem["type"] in ["badge", "sparkle", "price_tag"]:
            # 加载 PNG 素材
            asset_path = f"data/assets/{elem['type']}_{elem.get('style', 'default')}.png"
            asset = Image.open(asset_path).convert("RGBA")
            # 缩放 + 混合
            ...
```

---

#### 7. 文字排版"海报化"
**问题**：当前文字直接印在图上，缺乏设计感。

**修改文件**：`tools/text_renderer.py`

**解决方案**：
1. **字体升级**：引入阿里巴巴普惠体 Bold（`/fonts/AlibabaPuHuiTi-3/AlibabaPuHuiTi-3-Bold.ttf`）
2. **底部区块**：在文字区域生成 30% 高度的半透明黑色渐变底板
3. **高对比渲染**：标题使用亮黄色/纯白 + 黑色描边
4. **价格突出**：价格数字放大 1.5 倍

---

### 🟢 P3 - 低优先级（锦上添花）

#### 8. 装饰元素增强（投影 + 发光）
**问题**：装饰元素与背景在同一平面，不够突出。

**解决方案**：
- 贴图前先生成黑色遮罩，高斯模糊后偏移作为投影
- Sparkle 类使用 Add 混合模式叠加，产生发光效果

---

## 测试计划

### 样本数量
**10-15 组**（精简版），覆盖品类：
- 鞋子（运动鞋、休闲鞋）
- 包包（手提包、背包）
- 电子产品（手机、耳机）
- 服饰（外套、裤子）
- 配件（手表、眼镜）

### 测试用例
| ID | 图片 | 品类 | 预期背景场景 |
|----|------|------|--------------|
| TC-01 | xianyu_bag_01.jpg | 包包 | 云石台面/精品店 |
| TC-02 | xianyu_shoes_01.jpg | 球鞋 | 运动场地/街头 |
| TC-03 | shoes_02.jpg | 足球鞋 | 绿茵草地 |
| TC-04 | xiaohongshu_bag_01.jpg | 包包 | 精品店/家居 |

### 验证指标
1. **背景匹配度**：商品与背景场景是否合理
2. **透视一致性**：视角是否与原图一致
3. **文字可读性**：无方框、无重叠
4. **视觉层级**：装饰元素是否突出

---

## 实施顺序

### Phase 1：阻塞性修复
- [x] ~~P0-1 文字方框问题~~ → 等待实现
- [x] ~~P0-2 文字重叠问题~~ → 等待实现

### Phase 2：场景优化
- [ ] P1-3 场景-品类硬路由（SCENE_TEMPLATES）
- [ ] P1-4 VLM Prompt 场景+透视约束

### Phase 3：视觉升级
- [ ] P2-5 光影一致性
- [ ] P2-6 PNG 素材库
- [ ] P2-7 文字排版海报化

### Phase 4：细节打磨
- [ ] P3-8 装饰投影+发光

---

*最后更新: 2026-03-31*
*负责人: 李冠良、马在飞*

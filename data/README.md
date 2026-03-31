# 数据目录说明

## 目录结构

```
data/
├── original/          # 原始图片（来源截图）
├── input/             # 处理前的测试图片
├── intermediate/      # 调试用中间产物
└── output/            # 最终输出文件
```

## original/ - 原始图片

来源截图或随手拍的商品图片，每个图片对应一个同名 `.txt` 文件存放原始描述。

| 图片文件 | 商品 | 平台 |
|----------|------|------|
| xianyu_bag_01.jpg | 爱马仕Kelly包 | 闲鱼 |
| xianyu_shoes_01.jpg | 李宁音速11球鞋 | 闲鱼 |
| xianyu_shoes_02.txt | (配套描述) | 闲鱼 |
| xiaohongshu_bag_01.jpg | LV月亮包 | 小红书 |

## input/ - 测试图片

整理后的测试图片，用于批量评估。

## intermediate/ - 中间产物

调试用，存放处理过程中的临时文件：
- `_matted.png` - 抠图结果
- `_background.png` - SD生成背景
- `_fusion_*.png` - 融合过程图

## output/ - 最终输出

详见 [output/README.md](output/README.md)

---

## 添加新测试图片

1. 将图片放入 `original/` 目录
2. 创建同名 `.txt` 文件存放原始描述
3. 复制到 `input/` 目录进行测试

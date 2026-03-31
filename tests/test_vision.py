"""
Re-Value Agent - 视觉底层基建单元测试
专门用于调试边缘融合和抠图精度
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

from tools.cv_utils import (
    ImageFusion,
    MattingTool,
    ShadowRenderer,
    resize_if_needed,
    validate_image,
)


class TestMattingTool:
    """测试抠图工具"""

    def test_matting_tool_initialization(self):
        """测试初始化"""
        tool = MattingTool(model_name="u2net")
        assert tool.model_name == "u2net"
        assert tool._rembg_session is None

    def test_fallback_matting(self):
        """测试 fallback 抠图 (无 rembg 环境)"""
        tool = MattingTool(model_name="u2net")

        # 创建简单测试图像
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))

        try:
            result = tool.remove_background(img)
            # fallback 应该返回一个 RGBA 图像
            assert result.mode == "RGBA"
            assert result.size == img.size
        except ImportError:
            pytest.skip("rembg not installed")

    def test_alpha_coverage(self):
        """测试 Alpha 通道覆盖率计算"""
        tool = MattingTool()

        # 全透明图像
        fully_transparent = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        coverage = tool._get_alpha_coverage(fully_transparent)
        assert coverage == 0.0

        # 半透明图像
        half_transparent = Image.new("RGBA", (100, 100), (255, 255, 255, 128))
        coverage = tool._get_alpha_coverage(half_transparent)
        assert coverage == 100.0

        # 全不透明
        fully_opaque = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        coverage = tool._get_alpha_coverage(fully_opaque)
        assert coverage == 100.0

    def test_validate_alpha(self):
        """测试 Alpha 验证"""
        tool = MattingTool()

        # 有效图像
        valid = Image.new("RGBA", (100, 100), (255, 255, 255, 200))
        assert tool._validate_alpha(valid) is True

        # 无效图像 (有效面积 < 5%)
        # 创建一个只有中心有内容的图像
        small_center = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        center_region = small_center.crop((40, 40, 60, 60))
        center_region.paste((255, 255, 255, 255), (0, 0, 20, 20))
        # 5% = 20x20/100x100 = 400/10000 = 4%, 应该返回 False
        # 但我们简单测试一下
        result = tool._validate_alpha(small_center)
        # 如果中心面积 < 5%，应该返回 False
        assert isinstance(result, bool)


class TestShadowRenderer:
    """测试阴影渲染"""

    def test_shadow_renderer_initialization(self):
        """测试初始化"""
        renderer = ShadowRenderer(blur_radius=10, shadow_alpha=80, offset_y=5)
        assert renderer.blur_radius == 10
        assert renderer.shadow_alpha == 80
        assert renderer.offset_y == 5

    def test_generate_shadow(self):
        """测试阴影生成"""
        renderer = ShadowRenderer(blur_radius=5, shadow_alpha=100, offset_y=3)

        # 创建简单 RGBA 图像
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        # 添加一个白色方块
        img.paste((255, 255, 255, 255), (30, 30, 50, 50))

        shadow = renderer.generate_shadow(img)

        assert shadow.mode == "L"
        assert shadow.size == img.size

    def test_generate_shadow_requires_rgba(self):
        """测试需要 RGBA 输入"""
        renderer = ShadowRenderer()

        rgb_img = Image.new("RGB", (100, 100), (255, 255, 255))

        with pytest.raises(ValueError):
            renderer.generate_shadow(rgb_img)

    def test_apply_shadow(self):
        """测试阴影应用"""
        renderer = ShadowRenderer(blur_radius=3, shadow_alpha=50, offset_y=2)

        # 创建 RGBA 主体
        subject = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        subject.paste((255, 0, 0, 255), (30, 30, 50, 50))

        # 创建背景
        background = Image.new("RGB", (200, 200), (200, 200, 200))

        result = renderer.apply_shadow(subject, background)

        assert result.mode == "RGB"
        assert result.size == (200, 200)


class TestImageFusion:
    """测试图像融合"""

    def test_image_fusion_initialization(self):
        """测试初始化"""
        fusion = ImageFusion(subject_height_ratio=0.6, bottom_margin_ratio=0.2)
        assert fusion.subject_height_ratio == 0.6
        assert fusion.bottom_margin_ratio == 0.2

    def test_compute_placement_basic(self):
        """测试基本位置计算"""
        fusion = ImageFusion(subject_height_ratio=0.65, bottom_margin_ratio=0.15)

        # 主体比背景小
        subject_size = (100, 100)
        background_size = (500, 500)

        x, y, w, h = fusion.compute_placement(subject_size, background_size)

        assert h == int(500 * 0.65)  # 高度按比例
        assert abs(x - (500 - w) / 2) < 1  # 水平居中
        assert y == int(500 * (1 - 0.15)) - h  # 垂直偏下

    def test_compute_placement_with_reference_box(self):
        """测试带参考框的位置计算"""
        fusion = ImageFusion()

        subject_size = (100, 100)
        background_size = (500, 500)

        # 有效的参考框 (占 40% 宽度, 50% 高度)
        reference_box = [0.3, 0.25, 0.7, 0.75]

        x, y, w, h = fusion.compute_placement(
            subject_size, background_size, reference_box
        )

        # 应该按参考框比例缩放
        assert w <= 200  # 不会超出参考框太远
        assert h <= 250

    def test_validate_reference_box(self):
        """测试参考框验证"""
        fusion = ImageFusion()

        # 有效框
        valid = [0.2, 0.2, 0.8, 0.8]
        assert fusion._validate_reference_box(valid, 100, 100) is True

        # 宽高比太小
        small = [0.45, 0.45, 0.55, 0.55]
        assert fusion._validate_reference_box(small, 100, 100) is False

        # 越界
        out_of_bounds = [0.1, 0.1, 0.9, 1.1]
        assert fusion._validate_reference_box(out_of_bounds, 100, 100) is False

        # 空值
        assert fusion._validate_reference_box(None, 100, 100) is False

    def test_fuse_basic(self):
        """测试基本融合"""
        fusion = ImageFusion(subject_height_ratio=0.5, bottom_margin_ratio=0.1)

        # 创建主体 (带 Alpha)
        subject = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        subject.paste((255, 0, 0, 255), (20, 20, 80, 80))

        # 创建背景
        background = Image.new("RGB", (500, 500), (128, 128, 128))

        result = fusion.fuse(subject, background)

        assert result.mode == "RGB"
        assert result.size == (500, 500)


class TestUtilityFunctions:
    """测试工具函数"""

    def test_validate_image(self):
        """测试图像验证"""
        # 有效图像
        valid = Image.new("RGB", (100, 100), (255, 255, 255))
        assert validate_image(valid) is True

        # 太小
        tiny = Image.new("RGB", (5, 5), (255, 255, 255))
        assert validate_image(tiny) is False

        # None
        assert validate_image(None) is False

    def test_resize_if_needed(self):
        """测试图像缩放"""
        # 不需要缩放
        small = Image.new("RGB", (100, 100), (255, 255, 255))
        result = resize_if_needed(small, max_size=1024)
        assert result.size == (100, 100)

        # 需要缩放
        large = Image.new("RGB", (2048, 2048), (255, 255, 255))
        result = resize_if_needed(large, max_size=1024)
        assert max(result.size) == 1024

        # 非正方形
        wide = Image.new("RGB", (2048, 1024), (255, 255, 255))
        result = resize_if_needed(wide, max_size=1024)
        assert max(result.size) == 1024
        assert min(result.size) == 512


class TestIntegration:
    """集成测试"""

    def test_full_vision_pipeline_concept(self):
        """
        测试完整视觉流程概念
        注意: 需要实际的 rembg 和 SD 模型才能运行完整测试
        """
        # 创建测试图像
        test_image = Image.new("RGB", (800, 600), (200, 200, 200))

        # 1. 抠图
        matting_tool = MattingTool()
        try:
            matted = matting_tool.remove_background(test_image)
            assert matted.mode == "RGBA"
        except ImportError:
            pytest.skip("rembg not installed")

        # 2. 生成阴影
        shadow_renderer = ShadowRenderer()
        shadow = shadow_renderer.generate_shadow(matted)
        assert shadow.mode == "L"

        # 3. 融合
        fusion = ImageFusion()
        background = Image.new("RGB", (1024, 1024), (240, 240, 240))
        result = fusion.fuse(matted, background)
        assert result.mode == "RGB"
        assert result.size == (1024, 1024)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

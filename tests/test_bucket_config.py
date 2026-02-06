"""测试评分桶配置模块。"""

import pytest

from src.data_processing.bucket_config import (
    BucketConfig,
    find_bucket_for_score,
    get_all_bucket_configs,
    get_bucket_config,
)


class TestBucketConfig:
    """测试 BucketConfig 类。"""

    def test_contains_with_max_score(self):
        """测试有上限的评分区间。"""
        bucket = BucketConfig("test", 2.8, 3.0, 0.5)

        assert bucket.contains(2.8) is True
        assert bucket.contains(2.9) is True
        assert bucket.contains(2.99) is True
        assert bucket.contains(3.0) is False  # 右开区间
        assert bucket.contains(2.799) is False  # 低于下限
        assert bucket.contains(3.1) is False  # 高于上限

    def test_contains_without_max_score(self):
        """测试无上限的评分区间。"""
        bucket = BucketConfig("test", 4.0, None, 1.0)

        assert bucket.contains(4.0) is True
        assert bucket.contains(4.5) is True
        assert bucket.contains(5.0) is True
        assert bucket.contains(3.9) is False
        assert bucket.contains(100.0) is True  # 无上限

    def test_contains_with_float_precision(self):
        """测试浮点数精度处理。"""
        bucket = BucketConfig("test", 2.8, 3.0, 0.5)

        # 边界值考虑 epsilon
        assert bucket.contains(2.8 - 1e-10) is True
        assert bucket.contains(3.0 - 1e-10) is True

    def test_repr(self):
        """测试字符串表示。"""
        bucket = BucketConfig("2.8", 2.8, 3.0, 0.3)
        repr_str = repr(bucket)

        assert "BucketConfig" in repr_str
        assert "2.8" in repr_str
        assert "30%" in repr_str or "0.3" in repr_str


class TestDefaultBuckets:
    """测试默认评分桶配置。"""

    def test_get_all_bucket_configs(self):
        """测试获取所有默认桶配置。"""
        buckets = get_all_bucket_configs()

        assert len(buckets) == 4
        assert buckets[0].name == "2.8"
        assert buckets[1].name == "3.0"
        assert buckets[2].name == "3.5"
        assert buckets[3].name == "4.0"

    def test_get_bucket_config_valid(self):
        """测试获取有效的桶配置。"""
        bucket = get_bucket_config("3.0")

        assert bucket.name == "3.0"
        assert bucket.min_score == 3.0
        assert bucket.max_score == 3.5
        assert bucket.sampling_rate == 0.6

    def test_get_bucket_config_invalid(self):
        """测试获取无效的桶配置。"""
        with pytest.raises(ValueError, match="Unknown bucket name"):
            get_bucket_config("invalid")

    def test_find_bucket_for_score(self):
        """测试根据评分查找桶。"""
        bucket = find_bucket_for_score(2.8)
        assert bucket is not None and bucket.name == "2.8"
        bucket = find_bucket_for_score(2.9)
        assert bucket is not None and bucket.name == "2.8"

        bucket = find_bucket_for_score(3.0)
        assert bucket is not None and bucket.name == "3.0"
        bucket = find_bucket_for_score(3.4)
        assert bucket is not None and bucket.name == "3.0"

        bucket = find_bucket_for_score(3.5)
        assert bucket is not None and bucket.name == "3.5"
        bucket = find_bucket_for_score(3.9)
        assert bucket is not None and bucket.name == "3.5"

        bucket = find_bucket_for_score(4.0)
        assert bucket is not None and bucket.name == "4.0"
        bucket = find_bucket_for_score(5.0)
        assert bucket is not None and bucket.name == "4.0"

        assert find_bucket_for_score(2.7) is None

    def test_bucket_intervals_no_overlap(self):
        """测试评分区间不重叠。"""
        bucket = find_bucket_for_score(3.0)
        assert bucket is not None and bucket.name == "3.0"
        bucket = find_bucket_for_score(3.5)
        assert bucket is not None and bucket.name == "3.5"
        bucket = find_bucket_for_score(4.0)
        assert bucket is not None and bucket.name == "4.0"

    def test_sampling_rates(self):
        """测试采样率配置。"""
        buckets = get_all_bucket_configs()

        assert buckets[0].sampling_rate == 0.30
        assert buckets[1].sampling_rate == 0.60
        assert buckets[2].sampling_rate == 0.80
        assert buckets[3].sampling_rate == 1.0

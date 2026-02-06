"""测试评分桶配置模块。"""

import pytest

from src.data_processing.bucket_config import (
    BucketConfig,
    find_bucket_for_score,
    get_all_bucket_configs,
    get_bucket_config,
)


class TestBucketConfig:
    def test_contains_with_max_score(self):
        bucket = BucketConfig("test", 2.8, 3.0, 0.5)

        assert bucket.contains(2.8) is True
        assert bucket.contains(2.9) is True
        assert bucket.contains(3.0) is False
        assert bucket.contains(2.799) is False
        assert bucket.contains(3.1) is False

    def test_contains_without_max_score(self):
        bucket = BucketConfig("test", 4.0, None, 1.0)

        assert bucket.contains(4.0) is True
        assert bucket.contains(4.5) is True
        assert bucket.contains(5.0) is True
        assert bucket.contains(3.9) is False

    def test_contains_with_float_precision(self):
        bucket = BucketConfig("test", 2.8, 3.0, 0.5)

        assert bucket.contains(2.8 - 1e-10) is True
        assert bucket.contains(3.0 - 1e-10) is True

    def test_repr(self):
        bucket = BucketConfig("2.8", 2.8, 3.0, 0.3)
        repr_str = repr(bucket)

        assert "BucketConfig" in repr_str
        assert "2.8" in repr_str
        assert "30%" in repr_str or "0.3" in repr_str


class TestDefaultBuckets:
    def test_get_all_bucket_configs(self):
        buckets = get_all_bucket_configs()

        assert len(buckets) == 4
        assert buckets[0].name == "2.8"
        assert buckets[1].name == "3.0"
        assert buckets[2].name == "3.5"
        assert buckets[3].name == "4.0"

    def test_get_bucket_config_valid(self):
        bucket = get_bucket_config("3.0")

        assert bucket.name == "3.0"
        assert bucket.min_score == 3.0
        assert bucket.max_score == 3.5
        assert bucket.sampling_rate == 0.6

    def test_get_bucket_config_invalid(self):
        with pytest.raises(ValueError, match="Unknown bucket"):
            get_bucket_config("invalid")

    def test_find_bucket_for_score(self):
        bucket_28 = find_bucket_for_score(2.8)
        assert bucket_28 is not None and bucket_28.name == "2.8"
        bucket_29 = find_bucket_for_score(2.9)
        assert bucket_29 is not None and bucket_29.name == "2.8"
        bucket_30 = find_bucket_for_score(3.0)
        assert bucket_30 is not None and bucket_30.name == "3.0"
        bucket_34 = find_bucket_for_score(3.4)
        assert bucket_34 is not None and bucket_34.name == "3.0"
        bucket_35 = find_bucket_for_score(3.5)
        assert bucket_35 is not None and bucket_35.name == "3.5"
        bucket_39 = find_bucket_for_score(3.9)
        assert bucket_39 is not None and bucket_39.name == "3.5"
        bucket_40 = find_bucket_for_score(4.0)
        assert bucket_40 is not None and bucket_40.name == "4.0"
        bucket_50 = find_bucket_for_score(5.0)
        assert bucket_50 is not None and bucket_50.name == "4.0"
        assert find_bucket_for_score(2.7) is None

    def test_bucket_intervals_no_overlap(self):
        b30 = find_bucket_for_score(3.0)
        assert b30 is not None and b30.name == "3.0"
        b35 = find_bucket_for_score(3.5)
        assert b35 is not None and b35.name == "3.5"
        b40 = find_bucket_for_score(4.0)
        assert b40 is not None and b40.name == "4.0"

    def test_sampling_rates(self):
        buckets = get_all_bucket_configs()

        assert buckets[0].sampling_rate == 0.30
        assert buckets[1].sampling_rate == 0.60
        assert buckets[2].sampling_rate == 0.80
        assert buckets[3].sampling_rate == 1.0

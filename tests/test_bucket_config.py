import pytest

from src.data_processing.bucket_config import (
    BucketConfig,
    find_bucket_for_score,
    get_all_bucket_configs,
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

    def test_repr(self):
        bucket = BucketConfig("2.8", 2.8, 3.0, 0.3)
        repr_str = repr(bucket)
        assert "BucketConfig" in repr_str
        assert "2.8" in repr_str
        assert "30%" in repr_str or "0.3" in repr_str


class TestEnglishBuckets:
    def test_get_all_bucket_configs(self):
        buckets = get_all_bucket_configs("en")
        assert len(buckets) == 4
        assert buckets[0].name == "2.5"
        assert buckets[1].name == "3.0"
        assert buckets[2].name == "3.5"
        assert buckets[3].name == "4.0"

    @pytest.mark.parametrize(
        "score,expected",
        [
            (2.5, "2.5"),
            (2.9, "2.5"),
            (3.0, "3.0"),
            (3.4, "3.0"),
            (3.5, "3.5"),
            (3.9, "3.5"),
            (4.0, "4.0"),
            (5.0, "4.0"),
        ],
    )
    def test_find_bucket_for_score(self, score, expected):
        b = find_bucket_for_score(score, "en")
        assert b is not None and b.name == expected

    def test_find_bucket_for_score_out_of_range(self):
        assert find_bucket_for_score(2.4, "en") is None

    def test_bucket_intervals_no_overlap(self):
        b = find_bucket_for_score(3.0, "en")
        assert b is not None and b.name == "3.0"
        b = find_bucket_for_score(3.5, "en")
        assert b is not None and b.name == "3.5"
        b = find_bucket_for_score(4.0, "en")
        assert b is not None and b.name == "4.0"

    def test_sampling_rates(self):
        buckets = get_all_bucket_configs("en")
        assert buckets[0].sampling_rate == 0.25
        assert buckets[1].sampling_rate == 0.50
        assert buckets[2].sampling_rate == 0.80
        assert buckets[3].sampling_rate == 1.0


class TestChineseBuckets:
    def test_get_all_bucket_configs(self):
        buckets = get_all_bucket_configs("zh")
        assert len(buckets) == 4
        assert buckets[0].name == "2.5"
        assert buckets[1].name == "3.0"
        assert buckets[2].name == "3.5"
        assert buckets[3].name == "4.0"

    @pytest.mark.parametrize(
        "score,expected",
        [
            (2.5, "2.5"),
            (2.9, "2.5"),
            (3.0, "3.0"),
            (3.4, "3.0"),
            (3.5, "3.5"),
            (3.9, "3.5"),
            (4.0, "4.0"),
            (5.0, "4.0"),
        ],
    )
    def test_find_bucket_for_score(self, score, expected):
        b = find_bucket_for_score(score, "zh")
        assert b is not None and b.name == expected

    def test_find_bucket_for_score_out_of_range(self):
        assert find_bucket_for_score(2.4, "zh") is None

    def test_sampling_rates(self):
        buckets = get_all_bucket_configs("zh")
        assert buckets[0].sampling_rate == 0.40
        assert buckets[1].sampling_rate == 0.60
        assert buckets[2].sampling_rate == 0.90
        assert buckets[3].sampling_rate == 1.0

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


# 数据集配置：每种语言的特定属性
DATASET_CONFIGS = {
    "en": {
        "bucket_count": 4,
        "sampling_rates": [0.25, 0.50, 0.80, 1.0],
    },
    "zh": {
        "bucket_count": 4,
        "sampling_rates": [0.40, 0.60, 0.90, 1.0],
    },
}


@pytest.fixture(params=["en", "zh"])
def lang(request):
    return request.param


@pytest.fixture
def dataset_config(lang):
    return DATASET_CONFIGS[lang]


class TestBuckets:
    def test_get_all_bucket_configs(self, lang, dataset_config):
        buckets = get_all_bucket_configs(lang)
        assert len(buckets) == dataset_config["bucket_count"]
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
    def test_find_bucket_for_score(self, lang, score, expected):
        b = find_bucket_for_score(score, lang)
        assert b is not None and b.name == expected

    def test_find_bucket_for_score_out_of_range(self, lang):
        assert find_bucket_for_score(2.4, lang) is None

    def test_bucket_intervals_no_overlap(self, lang):
        b = find_bucket_for_score(3.0, lang)
        assert b is not None and b.name == "3.0"
        b = find_bucket_for_score(3.5, lang)
        assert b is not None and b.name == "3.5"
        b = find_bucket_for_score(4.0, lang)
        assert b is not None and b.name == "4.0"

    def test_sampling_rates(self, lang, dataset_config):
        buckets = get_all_bucket_configs(lang)
        expected_rates = dataset_config["sampling_rates"]
        for i, expected_rate in enumerate(expected_rates):
            assert buckets[i].sampling_rate == expected_rate

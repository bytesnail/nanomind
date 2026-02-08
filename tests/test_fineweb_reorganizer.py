"""Tests for fineweb_reorganizer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_processing.bucket_config import BucketConfig
from src.data_processing.fineweb_reorganizer import (
    create_pipeline,
    get_default_config,
    main,
    process_all_datasets,
    process_single_dataset,
    setup_logging,
)


class TestDefaults:
    @patch("src.data_processing.fineweb_reorganizer.get_processing_config")
    def test_defaults_loading(self, mock_get_processing_config):
        mock_get_processing_config.return_value = {
            "workers": 16,
            "tasks": 32,
            "random_seed": 123,
            "compression": "gzip",
            "max_file_size_bytes": 1024,
            "logging": {"format": "%(levelname)s - %(message)s"},
        }

        get_default_config.cache_clear()

        result = get_default_config()

        assert result["workers"] == 16
        assert result["tasks"] == 32
        assert result["random_seed"] == 123
        assert result["compression"] == "gzip"
        assert result["max_size"] == 1024
        assert result["log_format"] == "%(levelname)s - %(message)s"

    @patch("src.data_processing.fineweb_reorganizer.get_processing_config")
    def test_defaults_with_defaults(self, mock_get_processing_config):
        mock_get_processing_config.return_value = {}

        get_default_config.cache_clear()
        result = get_default_config()

        assert result["workers"] == 8
        assert result["tasks"] == 8
        assert result["random_seed"] == 42
        assert result["compression"] == "zstd"
        assert result["max_size"] == 512 * 1024 * 1024


class TestSetupLogging:
    def test_setup_logging_creates_handlers(self, tmp_path):
        log_dir = tmp_path / "logs"
        name = "test_logger"

        logger = setup_logging(log_dir, name)

        assert logger.name == f"fineweb_{name}"
        assert len(logger.handlers) >= 2
        assert (log_dir / name).exists()

    def test_setup_logging_returns_same_logger(self, tmp_path):
        log_dir = tmp_path / "logs"

        logger1 = setup_logging(log_dir, "test")
        logger2 = setup_logging(log_dir, "test")

        assert logger1 is logger2


class TestCreatePipeline:
    @patch("src.data_processing.fineweb_reorganizer.LocalPipelineExecutor")
    @patch("src.data_processing.fineweb_reorganizer.ParquetReader")
    @patch("src.data_processing.fineweb_reorganizer.ScoreFilter")
    @patch("src.data_processing.fineweb_reorganizer.BucketPathWriter")
    def test_create_pipeline_multi_bucket(
        self, mock_writer, mock_filter, mock_reader, mock_executor
    ):
        input_dir = Path("/input")
        output_dir = Path("/output")
        buckets = [
            BucketConfig("3.0", 3.0, 3.5, 0.6),
            BucketConfig("3.5", 3.5, 4.0, 0.8),
        ]

        create_pipeline(input_dir, output_dir, buckets, workers=4, tasks=8)

        mock_reader.assert_called_once()
        mock_filter.assert_called_once()
        mock_writer.assert_called_once()
        mock_executor.assert_called_once()

        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["tasks"] == 8
        assert call_kwargs["workers"] == 4


class TestProcessSingleDataset:
    @patch("src.data_processing.fineweb_reorganizer.create_pipeline")
    @patch("src.data_processing.fineweb_reorganizer.setup_logging")
    def test_process_single_dataset(
        self, mock_setup_logging, mock_create_pipeline, tmp_path
    ):
        # Suppress unused mock warning
        assert mock_setup_logging is not None

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        buckets = [BucketConfig("3.0", 3.0, 3.5, 0.6)]
        mock_pipeline = MagicMock()
        mock_create_pipeline.return_value = mock_pipeline

        result = process_single_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            buckets=buckets,
            workers=2,
            tasks=2,
        )

        assert result == ["3.0"]
        mock_create_pipeline.assert_called_once()
        mock_pipeline.run.assert_called_once()


class TestProcessAllDatasets:
    @patch("src.data_processing.fineweb_reorganizer.process_single_dataset")
    @patch("src.data_processing.fineweb_reorganizer.get_all_bucket_configs")
    @patch("src.data_processing.fineweb_reorganizer.get_dataset_configs")
    def test_process_all_datasets(
        self, mock_get_dataset_configs, mock_get_buckets, mock_process_single
    ):
        mock_get_dataset_configs.return_value = {
            "en": {
                "name": "fineweb_edu_en",
                "input_dir": "/tmp/en_input",
                "output_dir": "/tmp/en_output",
            },
        }

        mock_get_buckets.return_value = [BucketConfig("3.0", 3.0, 3.5, 0.6)]
        mock_process_single.return_value = ["3.0"]

        with patch.object(Path, "exists", return_value=True):
            results = process_all_datasets()

        assert "en" in results
        assert results["en"] == ["3.0"]

    @patch("src.data_processing.fineweb_reorganizer.get_dataset_configs")
    def test_process_all_datasets_skips_missing_input(
        self, mock_get_dataset_configs, tmp_path
    ):
        mock_get_dataset_configs.return_value = {
            "en": {
                "name": "fineweb_edu_en",
                "input_dir": str(tmp_path / "nonexistent"),
                "output_dir": str(tmp_path / "output"),
            },
        }

        results = process_all_datasets()
        assert results == {}


class TestMain:
    @patch("src.data_processing.fineweb_reorganizer.process_all_datasets")
    def test_main_success(self, mock_process_all):
        mock_process_all.return_value = {"en": ["3.0", "3.5"], "zh": ["2.0", "3.0"]}

        result = main()

        assert result == 0
        mock_process_all.assert_called_once()

    @patch("src.data_processing.fineweb_reorganizer.process_all_datasets")
    def test_main_failure(self, mock_process_all):
        mock_process_all.side_effect = Exception("Test error")

        result = main()

        assert result == 1


class TestIntegration:
    def test_end_to_end_with_real_config(self):
        from src.data_processing.config_loader import (
            get_dataset_configs,
            get_raw_bucket_configs,
        )

        datasets = get_dataset_configs()

        assert "en" in datasets or "zh" in datasets

        for lang in datasets:
            buckets = get_raw_bucket_configs(lang)
            assert len(buckets) > 0

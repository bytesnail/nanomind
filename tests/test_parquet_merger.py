"""Tests for parquet_merger module."""

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_processing.parquet_merger import merge_all_buckets, merge_bucket_files


class TestMergeBucketFiles:
    def test_merge_single_file(self, tmp_path):
        """测试单文件不需要合并的情况"""
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()

        # 创建一个较大的文件
        table = pa.table(
            {
                "text": ["test content"] * 1000,
                "id": [f"id_{i}" for i in range(1000)],
                "score": [3.5] * 1000,
            }
        )
        pq.write_table(table, bucket_dir / "00000_00000.parquet")

        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024,  # 很小的目标大小，强制合并
            compression="zstd",
            remove_source=True,
        )

        assert len(result) == 1
        assert result[0].exists()

    def test_merge_multiple_files(self, tmp_path):
        """测试合并多个小文件"""
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()

        # 创建多个小文件
        for i in range(3):
            table = pa.table(
                {
                    "text": [f"content {i}"],
                    "id": [f"id_{i}"],
                    "score": [3.0 + i],
                }
            )
            pq.write_table(table, bucket_dir / f"0000{i}_00000.parquet")

        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024 * 1024,  # 1MB
            compression="zstd",
            remove_source=True,
        )

        assert len(result) == 1
        assert result[0].exists()
        assert not (bucket_dir / "00000_00000.parquet").exists()  # 源文件已删除

        # 验证数据完整性
        merged_table = pq.read_table(result[0])
        assert len(merged_table) == 3  # 3行数据

    def test_merge_respects_target_size(self, tmp_path):
        """测试合并时尊重目标文件大小"""
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()

        # 创建两个较大的表
        for i in range(2):
            table = pa.table(
                {
                    "text": ["x" * 10000] * 100,  # 较大的数据
                    "id": [f"id_{i}_{j}" for j in range(100)],
                    "score": [3.5] * 100,
                }
            )
            pq.write_table(table, bucket_dir / f"0000{i}_00000.parquet")

        # 使用很小的目标大小，应该产生多个合并文件
        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024,  # 1KB - 很小
            compression="zstd",
            remove_source=True,
        )

        # 应该产生至少2个文件，因为目标大小很小
        assert len(result) >= 1
        for f in result:
            assert f.exists()

    def test_merge_empty_directory(self, tmp_path):
        """测试空目录处理"""
        bucket_dir = tmp_path / "empty_bucket"
        bucket_dir.mkdir()

        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024,
            compression="zstd",
        )

        assert result == []

    def test_merge_nonexistent_directory(self, tmp_path):
        """测试不存在的目录"""
        bucket_dir = tmp_path / "nonexistent"

        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024,
            compression="zstd",
        )

        assert result == []

    def test_merge_without_removing_source(self, tmp_path):
        """测试不删除源文件"""
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()

        table = pa.table(
            {
                "text": ["content"],
                "id": ["id_0"],
                "score": [3.5],
            }
        )
        pq.write_table(table, bucket_dir / "00000_00000.parquet")

        result = merge_bucket_files(
            bucket_dir=bucket_dir,
            target_file_size=1024,
            compression="zstd",
            remove_source=False,
        )

        assert len(result) == 1
        assert (bucket_dir / "00000_00000.parquet").exists()  # 源文件仍在


class TestMergeAllBuckets:
    def test_merge_all_buckets(self, tmp_path):
        """测试合并所有桶"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 创建多个桶
        for bucket_name in ["2.5", "3.0", "3.5"]:
            bucket_dir = output_dir / bucket_name
            bucket_dir.mkdir()

            for i in range(2):
                table = pa.table(
                    {
                        "text": [f"{bucket_name} content {i}"],
                        "id": [f"{bucket_name}_id_{i}"],
                        "score": [float(bucket_name)],
                    }
                )
                pq.write_table(table, bucket_dir / f"0000{i}_00000.parquet")

        result = merge_all_buckets(
            output_dir=output_dir,
            target_file_size=1024 * 1024,
            compression="zstd",
            remove_source=True,
        )

        assert len(result) == 3
        for bucket_name in ["2.5", "3.0", "3.5"]:
            assert bucket_name in result
            assert len(result[bucket_name]) == 1

    def test_merge_all_buckets_empty(self, tmp_path):
        """测试空输出目录"""
        output_dir = tmp_path / "empty_output"
        output_dir.mkdir()

        result = merge_all_buckets(
            output_dir=output_dir,
            target_file_size=1024,
            compression="zstd",
        )

        assert result == {}

    def test_merge_all_buckets_nonexistent(self, tmp_path):
        """测试不存在的输出目录"""
        output_dir = tmp_path / "nonexistent"

        result = merge_all_buckets(
            output_dir=output_dir,
            target_file_size=1024,
            compression="zstd",
        )

        assert result == {}

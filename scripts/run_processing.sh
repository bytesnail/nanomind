#!/bin/bash
# FineWeb-Edu 数据集重组批量运行脚本
# 用于生产环境全量数据处理

set -e  # 遇到错误立即退出

# 默认配置
INPUT_DIR="data/datasets/HuggingFaceFW/fineweb-edu/data"
OUTPUT_DIR="data/datasets/fineweb/en"
WORKERS=8
PARALLEL_BUCKETS=1
SEED=42
COMPRESSION="zstd"
MAX_FILE_SIZE=$((512 * 1024 * 1024))  # 512MB

# 日志目录
LOG_DIR="logs/fineweb_processing"
mkdir -p "$LOG_DIR"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --parallel-buckets)
            PARALLEL_BUCKETS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --compression)
            COMPRESSION="$2"
            shift 2
            ;;
        --help)
            echo "FineWeb-Edu 数据重组批量运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --workers N          每个桶的 worker 数量（默认：8）"
            echo "  --parallel-buckets N 同时运行的桶数量（默认：1）"
            echo "  --seed N             随机种子（默认：42）"
            echo "  --input DIR          源数据目录"
            echo "  --output DIR         输出目录"
            echo "  --compression TYPE   压缩格式：zstd, gzip, snappy（默认：zstd）"
            echo "  --help               显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0"
            echo "  $0 --workers 16 --parallel-buckets 2"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查依赖
echo "检查依赖..."
python -c "import datatrove" 2>/dev/null || {
    echo "错误: datatrove 未安装"
    exit 1
}

python -c "import pybloom_live" 2>/dev/null || {
    echo "警告: pybloom-live 未安装，去重功能将不可用"
}

# 检查输入目录
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "=============================================="
echo "FineWeb-Edu 数据重组开始"
echo "=============================================="
echo "开始时间: $(date)"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Workers: $WORKERS"
echo "并行桶数: $PARALLEL_BUCKETS"
echo "随机种子: $SEED"
echo "压缩格式: $COMPRESSION"
echo "=============================================="

# 运行处理
python -m src.data_processing.fineweb_reorganizer \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --parallel-buckets "$PARALLEL_BUCKETS" \
    --compression "$COMPRESSION" \
    --max-file-size "$MAX_FILE_SIZE" \
    2>&1 | tee "$LOG_DIR/processing_$(date +%Y%m%d_%H%M%S).log"

# 检查处理结果
EXIT_CODE=${PIPESTATUS[0]}

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_HOURS=$((DURATION / 3600))
DURATION_MINUTES=$(((DURATION % 3600) / 60))

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "=============================================="
    echo "处理完成"
    echo "结束时间: $(date)"
    echo "耗时: ${DURATION_HOURS}小时 ${DURATION_MINUTES}分钟"
    echo "=============================================="
    
    # 运行验证
    echo ""
    echo "运行验证..."
    python scripts/validate_output.py --input "$OUTPUT_DIR" --verbose
    
    exit 0
else
    echo "=============================================="
    echo "处理失败，退出码: $EXIT_CODE"
    echo "结束时间: $(date)"
    echo "=============================================="
    exit $EXIT_CODE
fi

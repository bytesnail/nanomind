#!/bin/bash
# Nanomind 预训练启动脚本
# 支持 DeepSpeed 和 Accelerate

set -e

# 默认配置
CONFIG_FILE="configs/nanomind_1b.yaml"
NUM_GPUS=2
USE_DEEPSPEED=true
USE_ACCELERATE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed)
            USE_DEEPSPEED=true
            USE_ACCELERATE=false
            shift
            ;;
        --accelerate)
            USE_ACCELERATE=true
            USE_DEEPSPEED=false
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config PATH       配置文件路径 (默认: configs/nanomind_1b.yaml)"
            echo "  --gpus N            GPU 数量 (默认: 2)"
            echo "  --deepspeed         使用 DeepSpeed (默认)"
            echo "  --accelerate        使用 Accelerate"
            echo "  --help              显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# 设置环境变量
export WANDB_PROJECT="${WANDB_PROJECT:-nanomind-pretraining}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=8

echo "========================================"
echo "Nanomind 预训练"
echo "========================================"
echo "配置文件: $CONFIG_FILE"
echo "GPU 数量: $NUM_GPUS"
echo "启动方式: $([ "$USE_ACCELERATE" = true ] && echo "Accelerate" || echo "DeepSpeed")"
echo "========================================"

# 创建输出目录
OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['output_dir'])")
mkdir -p "$OUTPUT_DIR"

# 启动训练
if [ "$USE_ACCELERATE" = true ]; then
    echo "使用 Accelerate 启动..."
    
    # 检查 accelerate 配置
    if [ ! -f "configs/accelerate_config.yaml" ]; then
        echo "错误: configs/accelerate_config.yaml 不存在"
        exit 1
    fi
    
    accelerate launch \
        --config_file configs/accelerate_config.yaml \
        --num_processes "$NUM_GPUS" \
        scripts/train.py \
        --config "$CONFIG_FILE"
        
elif [ "$USE_DEEPSPEED" = true ]; then
    echo "使用 DeepSpeed 启动..."
    
    # 获取 DeepSpeed 配置
    DS_CONFIG=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['deepspeed'])")
    
    if [ ! -f "$DS_CONFIG" ]; then
        echo "错误: DeepSpeed 配置文件不存在: $DS_CONFIG"
        exit 1
    fi
    
    deepspeed \
        --num_gpus="$NUM_GPUS" \
        scripts/train.py \
        --config "$CONFIG_FILE" \
        --deepspeed "$DS_CONFIG"
else
    echo "错误: 未指定启动方式"
    exit 1
fi

echo "========================================"
echo "训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

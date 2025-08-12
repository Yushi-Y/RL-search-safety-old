#!/bin/bash

# Weight Difference Analysis Runner Script

echo "🚀 Starting Model Weight Difference Analysis"
echo "=============================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    DEVICE="cuda"
else
    echo "⚠️  CUDA not detected, using CPU"
    DEVICE="cpu"
fi

# Create results directory
mkdir -p model_diffing_results

# Run the analysis
python model_diffing/weight_difference.py \
    --original_model "Qwen/Qwen2.5-7B-Instruct" \
    --rl_model "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo" \
    --save_dir "./model_diffing_results" \
    --device "$DEVICE"

echo ""
echo "🎉 Analysis completed!"
echo "📁 Results saved to: ./model_diffing_results/"
echo "📊 Check the generated visualizations and reports for detailed insights." 
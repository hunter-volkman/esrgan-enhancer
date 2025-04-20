#!/bin/bash
# Name: scripts/prepare_training.sh
# Purpose: Copy datasets and prepare for training

# Create directories if they don't exist
mkdir -p /workspace/datasets/custom/train/hr
mkdir -p /workspace/datasets/custom/val/hr
mkdir -p /workspace/datasets/custom/val/lr
mkdir -p /workspace/options

# Copy training data
cp -r /workspace/data/datasets/custom/train/hr/* /workspace/datasets/custom/train/hr/ 2>/dev/null || echo "Warning: No training images found"
cp /workspace/data/datasets/custom/train/meta_info.txt /workspace/datasets/custom/train/ 2>/dev/null || echo "Warning: No meta_info.txt found"

# Copy validation data
cp -r /workspace/data/datasets/custom/val/hr/* /workspace/datasets/custom/val/hr/ 2>/dev/null || echo "Warning: No validation HR images found"
cp -r /workspace/data/datasets/custom/val/lr/* /workspace/datasets/custom/val/lr/ 2>/dev/null || echo "Warning: No validation LR images found"

# Copy configuration file
cp /workspace/data/options/finetune_security_cam.yml /workspace/options/ 2>/dev/null || echo "Warning: No finetune_security_cam.yml found"

# Optimize for 8GB VRAM (RTX 3070 Ti)
if [ -f "/workspace/options/finetune_security_cam.yml" ]; then
    sed -i 's/gt_size: 256/gt_size: 192/' /workspace/options/finetune_security_cam.yml 2>/dev/null
    sed -i 's/batch_size_per_gpu: 4/batch_size_per_gpu: 2/' /workspace/options/finetune_security_cam.yml 2>/dev/null
    echo "✅ Config optimized for 8GB VRAM"
else
    echo "❌ Config file not found"
fi

echo "✅ Training data preparation complete"

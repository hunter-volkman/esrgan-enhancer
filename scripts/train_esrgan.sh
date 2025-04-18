#!/bin/bash
CONFIG="options/finetune_security_cam.yml"

echo "Starting fine-tuning with config: $CONFIG"
python /workspace/Real-ESRGAN/BasicSR/basicsr/train.py -opt "$CONFIG" --auto_resume

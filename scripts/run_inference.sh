#!/bin/bash
INPUT_DIR=${1:-"./test_images"}
OUTPUT_DIR=${2:-"./results"}
MODEL="RealESRGAN_x4plus"
TILE=512
OUTSCALE=3

echo "Running batch inference on $INPUT_DIR -> $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

for IMAGE in "$INPUT_DIR"/*.{jpg,jpeg,png}; do
    if [ -f "$IMAGE" ]; then
        echo "Processing: $IMAGE"
        python inference_realesrgan.py \
            -n $MODEL \
            -i "$IMAGE" \
            -o "$OUTPUT_DIR" \
            --tile $TILE \
            --outscale $OUTSCALE
    fi
done

echo "✓ Inference completed."

#!/usr/bin/env bash
set -e

WORKSPACE="/home/sharkum/Desktop/Vision/Assignment 3"
PYTHON="$WORKSPACE/.env/tadgs/bin/python"
cd "$WORKSPACE/TAD-GS"

rm -rf "$WORKSPACE/runs/smoke_test"

for K in 8 4 2; do
    MODEL_DIR="$WORKSPACE/runs/tadgs_k${K}"
    if [ -d "$MODEL_DIR" ]; then
        if [ -f "$MODEL_DIR/point_cloud/best/point_cloud.ply" ] && grep -q "Iteration: 8000" "$MODEL_DIR/training_log.txt" 2>/dev/null; then
            echo "[Skip] k=$K already trained to completion."
            continue
        else
            echo "[Clean] Removing incomplete previous run at $MODEL_DIR"
            rm -rf "$MODEL_DIR"
        fi
    fi
    echo "============================================================"
    echo "Starting TAD-GS Training for frame_gap k=$K"
    echo "============================================================"
    "$PYTHON" train.py --quiet --eval --vad --tat --tow \
        --configpath arguments/neural_3d_video/coffee_martini.json \
        --model_path "$MODEL_DIR" \
        --source_path "$WORKSPACE/data/neural_3d_video/scenes/coffee_martini/colmap_0" \
        --duration 50 \
        --frame_gap "$K" \
        --flow_masks_path "$WORKSPACE/data/neural_3d_video/flow_masks/coffee_martini/valid_masks"
done

cd "$WORKSPACE"
echo "============================================================"
echo "Running Complete Evaluation & Generating Plots/Visuals"
echo "============================================================"
"$PYTHON" scripts/evaluate_all.py
"$PYTHON" scripts/generate_plots_and_visuals.py
echo "ALL_EXPERIMENTS_DONE"


#!/bin/bash
echo "=== colmap_pipeline.py ==="
cat /home/sharkum/Desktop/Vision/Assignment2/colmap_pipeline.py
echo "=== test_env/bin ==="
ls /home/sharkum/Desktop/Vision/Assignment2/test_env/bin 2>/dev/null | head -n 40
echo "=== talent_env/bin ==="
ls /home/sharkum/.local/share/mamba/envs/talent_env/bin 2>/dev/null | grep -E "colmap|python|nvcc|pip"

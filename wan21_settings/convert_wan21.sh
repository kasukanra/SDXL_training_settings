#!/usr/bin/env bash
set -euo pipefail

# 1) Activate your ai-toolkit venv
cd /home/pure_water_100/workspace/ai-toolkit
source venv/bin/activate

# 2) Where your numbered Wan21 transformer dirs live:
INPUT_BASE="/home/pure_water_100/aitk_models/wan21_fantasy_art_01/fantasy_art_wan21_finetune_v1"
FOLDER_PREFIX="$(basename "$INPUT_BASE")"

# 3) Where to write your converted safetensors:
OUTPUT_BASE="/home/pure_water_100/model_store/wan21_fft/fantasy_art/fantasy_art_01"

# 5) Where to drop Symlinks for ComfyUI
SYMLINK_DIR="/home/pure_water_100/ComfyUI/models/diffusion_models/wan_fft/fantasy_art_01"
mkdir -p "$SYMLINK_DIR"

mkdir -p "$OUTPUT_BASE"

# 6) Step range & interval
START_STEP=5000
END_STEP=60000
STEP_INTERVAL=1000

echo "📂 INPUT_BASE:    $INPUT_BASE"
echo "📁 PREFIX:        $FOLDER_PREFIX"
echo "📥 OUTPUT_BASE:   $OUTPUT_BASE"
echo "🔗 SYMLINK_DIR:   $SYMLINK_DIR"
echo "🔢 Steps: $START_STEP → $END_STEP by $STEP_INTERVAL"

for (( step=$START_STEP; step<=END_STEP; step+=STEP_INTERVAL )); do
    # e.g. fantasy_art_wan21_finetune_v1_000020800
    folder_name=$(printf "%s_%09d" "$FOLDER_PREFIX" "$step")
    snapshot_dir="${INPUT_BASE}/${folder_name}/transformer"
    index_json="${snapshot_dir}/diffusion_pytorch_model.safetensors.index.json"
    output_path="${OUTPUT_BASE}/${step}.safetensors"
    symlink_path="${SYMLINK_DIR}/${step}.safetensors"

    if [[ -d "$snapshot_dir" && -f "$index_json" ]]; then
        echo "➡️  Converting ${folder_name}/transformer → ${step}.safetensors"
        python /home/pure_water_100/workspace/test_scripts/wan21/convert_wan21_to_comfy.py \
            "$snapshot_dir" \
            "$index_json" \
            "$output_path"

        if [[ -f "$output_path" ]]; then
            ln -sf "$output_path" "$symlink_path"
            echo "   🔗 Symlinked → $(basename "$symlink_path")"
        else
            echo "   ⚠ Conversion failed for step $step"
        fi
    else
        echo "🚫  Missing: ${snapshot_dir} or index.json"
    fi
done

echo "✅ All done!"
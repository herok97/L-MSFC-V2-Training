#!/bin/bash
ROOT_DIR=$1
NUM_MAX_PROCESS=$2

INPUT_VIDEOS=("pexels_oppv_p1" "pixabay" "pixabay_crowded" "videos_crowded" "videos_mppv_p1_13")
OUTPUT_DIR=${ROOT_DIR}/extracted_frames

mkdir -p ${OUTPUT_DIR}

for VIDEO in "${INPUT_VIDEOS[@]}"; do
    for file in ${ROOT_DIR}/${VIDEO}/*.mp4; do
        filename=$(basename -- "$file")
        extension=${filename##*.}
        filename_without_extension=${filename%.*}
        
        ffmpeg -i ${file} -vf select=not(mod(n\,3)) -vsync 0 -pix_fmt rgb24 ${OUTPUT_DIR}/${filename_without_extension}_%d.png &

        active_process_count=$(jobs | wc -l)
        if [ "$active_process_count" -ge "${NUM_MAX_PROCESS}" ]; then
            wait  # Wating for maximum processes
        fi
    done
done

wait
echo "Finished"
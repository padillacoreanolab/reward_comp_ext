#!/bin/bash
project_dir=/scratch/back_up/reward_competition_extention

cd ${experiment_dir}

video_directory=${project_dir}/data/rce_cohort_3
output_directory=/scratch/back_up/reward_competition_extention/temp/reencoded_videos

for full_path in ${video_directory}/*/*/*.h264; do
    echo Currently working on: ${full_path}
    file_name=${full_path##*/}
    base_name="${file_name%.h264}"

    echo Directory: ${dir_name}
    echo File Name: ${file_name}
    echo Base Name: ${base_name}

    mp4_out_path=${output_directory}/${base_name}.original.mp4
    echo Converting h264 to mp4
    
    ffmpeg -n \
    -i ${full_path} \
    -filter:v \
    "scale=1280:720:flags=lanczos" \
    -c:a copy ${mp4_out_path}

    echo Reencoding mp4
    ffmpeg -n -i ${mp4_out_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 ${output_directory}/${base_name}.fixed.mp4

done

echo All Done!



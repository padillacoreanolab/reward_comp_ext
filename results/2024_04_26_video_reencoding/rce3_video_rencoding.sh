#!/bin/bash
project_dir=/scratch/back_up/reward_competition_extention

cd ${experiment_dir}

video_directory=${project_dir}/data/rce_cohort_3
output_directory=/scratch/back_up/reward_competition_extention/in_progress/rce3/reencoded_videos

for full_path in ${video_directory}/*/*/*.h264; do
    echo Currently working on: ${full_path}
    file_name=${full_path##*/}
    base_name="${file_name%.h264}"
    mp4_out_path=${output_directory}/${base_name}.fixed.mp4

    echo Directory: ${dir_name}
    echo File Name: ${file_name}
    echo Base Name: ${base_name}
    echo Converting h264 to mp4
    echo Reencoding mp4
    
    ffmpeg -n -i ${full_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 ${mp4_out_path}

done

echo All Done!



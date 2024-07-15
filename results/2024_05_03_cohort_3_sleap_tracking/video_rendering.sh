#!/bin/bash
project_dir=/scratch/back_up/reward_competition_extention
experiment_dir=${project_dir}

cd ${experiment_dir}

video_directory=${experiment_dir}/final_proc/id_corrected/rce3

for full_path in ${video_directory}/*/*id_corrected.slp; do
    echo Currently working on: ${full_path}
    sleap-render ${full_path}
done

echo All Done!



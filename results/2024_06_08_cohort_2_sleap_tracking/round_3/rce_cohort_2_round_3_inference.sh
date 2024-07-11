#!/bin/bash

# Training
round_number=round_3
project_dir=/nancy/projects/reward_competition_extention/results/2024_03_25_rce3_preprocessing
cd ${project_dir}

cd ${experiment_dir}

model_directory=/nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_3/models/rce3_round_3_baseline_medium_rf.bottomup

video_directory=/scratch/back_up/reward_competition_extention/final_proc/reencoded_videos/rce_cohort_2
output_directory=/scratch/back_up/reward_competition_extention/in_progress/rce3/sleap/round_3
# Process function
track_with_sleap() {
    input_file=$1
    output_file=$2
    number_of_subjects=$3

    # Check if output file already exists
    if [ -f "${output_file}" ]; then
        echo "Output file ${output_file} already exists, skipping..."

    else
        echo "Processing ${input_file}..."

        sleap-track ${input_file} --tracking.tracker simplemaxtracks \
        --tracking.similarity iou \
        --tracking.match greedy \
        --batch_size 1 \
        --max_instances ${number_of_subjects} \
        --tracking.clean_instance_count ${number_of_subjects} \
        --tracking.target_instance_count ${number_of_subjects} \
        -m ${model_directory} \
        -o ${output_file} \
        --tracking.max_tracking 1 \
        --tracking.max_tracks ${number_of_subjects}

    fi

    echo "Input:" >> sleap_tracked_files.txt
    echo "${input_file}" >> sleap_tracked_files.txt
    echo "Output:" >> sleap_tracked_files.txt
    echo "${output_file}" >> sleap_tracked_files.txt
}

for full_path in ${video_directory}/*2023*/*1.fixed.mp4; do
    echo "Currently starting: ${full_path}"

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"
    recording_name=${base_name%%.*}
    
    recording_dir=${output_directory}/${recording_name}
    mkdir -p ${recording_dir}

    # Tracking with 2 subject
    total_subjects=2

    # Replace this with how you form your output file name
    output_file=${recording_dir}/${base_name}.${total_subjects}_subj.${round_number}.predicted_frames.slp
    track_with_sleap ${full_path} ${output_file} ${total_subjects} 

done

for full_path in ${video_directory}/*2023*/*2.fixed.mp4; do
    echo "Currently starting: ${full_path}"

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"
    recording_name=${base_name%%.*}
    
    recording_dir=${output_directory}/${recording_name}
    mkdir -p ${recording_dir}

    # Tracking with 2 subject
    total_subjects=2

    # Replace this with how you form your output file name
    output_file=${recording_dir}/${base_name}.${total_subjects}_subj.${round_number}.predicted_frames.slp
    track_with_sleap ${full_path} ${output_file} ${total_subjects} 

done

for full_path in ${video_directory}/*2023*/*1.fixed.mp4; do
    echo "Currently starting: ${full_path}"

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"
    recording_name=${base_name%%.*}
    
    recording_dir=${output_directory}/${recording_name}
    mkdir -p ${recording_dir}

    # Tracking with 1 subject
    total_subjects=1

    # Replace this with how you form your output file name
    output_file=${recording_dir}/${base_name}.${total_subjects}_subj.${round_number}.predicted_frames.slp
    track_with_sleap ${full_path} ${output_file} ${total_subjects} 

done

for full_path in ${video_directory}/*2023*/*2.fixed.mp4; do
    echo "Currently starting: ${full_path}"

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"
    recording_name=${base_name%%.*}
    
    recording_dir=${output_directory}/${recording_name}
    mkdir -p ${recording_dir}

    # Tracking with 1 subject
    total_subjects=1

    # Replace this with how you form your output file name
    output_file=${recording_dir}/${base_name}.${total_subjects}_subj.${round_number}.predicted_frames.slp
    track_with_sleap ${full_path} ${output_file} ${total_subjects} 

done

echo All Done!
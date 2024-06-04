#!/bin/bash

# Training
round_number=round_1
project_dir=/nancy/projects/reward_competition_extention/results/2024_03_25_rce3_preprocessing
cd ${project_dir}

cd ${experiment_dir}

model_directory=/scratch/back_up/reward_competition_extention/temp/rce3_round_1_baseline_medium_rf.bottomup

video_directory=/scratch/back_up/reward_competition_extention/data/rce_cohort_3
output_directory=/scratch/back_up/reward_competition_extention/temp/sleap
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

for full_path in ${video_directory}/*/*/*fixed.mp4; do
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

    # Tracking with 2 subjects
    total_subjects=2

    # Replace this with how you form your output file name
    output_file=${recording_dir}/${base_name}.${total_subjects}_subj.${round_number}.predicted_frames.slp
    track_with_sleap ${full_path} ${output_file} ${total_subjects} 

done

echo All Done!
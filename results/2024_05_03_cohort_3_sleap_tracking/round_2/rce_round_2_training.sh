#!/bin/bash

# Training
echo "training"
project_dir=/nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_2
cd ${project_dir}

sleap-train /nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_2/rce3_round_2_baseline_medium_rf.bottomup.json /scratch/back_up/reward_competition_extention/final_proc/sleap_labeling/rce3/round_1/pkg_slp/2024_05_20.labeled_frames.fixed.pkg.pkg.pkg.slp

echo All Done!
#!/bin/bash

# Training
echo "training"
project_dir=/blue/npadillacoreano/ryoi360/projects/reward_comp/repos/reward_comp_ext
cd ${project_dir}

sleap-train ${project_dir}/results/2024_05_03_cohort_3_sleap_tracking/rce3_round_1_baseline_medium_rf.bottomup.json ${project_dir}/data/sleap/2024_05_07.labeled_frames.fixed.pkg.pkg.slp

echo All Done!
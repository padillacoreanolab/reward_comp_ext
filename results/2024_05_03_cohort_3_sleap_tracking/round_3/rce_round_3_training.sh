#!/bin/bash

# Training
echo "training"
project_dir=/nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_3
cd ${project_dir}

sleap-train /nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_3/rce3_round_3_baseline_medium_rf.bottomup.json /nancy/user/riwata/projects/reward_comp_ext/results/2024_05_03_cohort_3_sleap_tracking/round_3/rce3_round_3.fixed_frames.pkg.pkg.pkg.slp

echo All Done!
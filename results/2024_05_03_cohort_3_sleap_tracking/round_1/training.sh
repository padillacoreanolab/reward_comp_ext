#!/bin/bash

# Training
echo "training"
project_dir=/nancy/projects/reward_competition_ephys_analysis_with_omission_and_divider_controls/results/2023_01_12_rc_sleap
cd ${project_dir}

sleap-train ${project_dir}/proc/sleap_labels/baseline_medium_rf.bottomup.json ${project_dir}/proc/sleap_labels/rc_om_and_comp_combined.fixed.mp4.pkg.slp

echo All Done!
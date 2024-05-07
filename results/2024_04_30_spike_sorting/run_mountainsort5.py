#!/usr/bin/env python3
"""
"""
import os
import time
import glob
import warnings
import numpy as np

import matplotlib.pyplot as plt

import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
from spikeinterface.exporters import export_to_phy

import mountainsort5 as ms5

from probeinterface import read_prb

# Reading in the probe file
probe_object = read_prb("/nancy/projects/reward_competition_extention/data/linear_probe_with_large_spaces.prb")

# Looking up all the recording files
recording_filepath_glob = "/scratch/back_up/reward_competition_extention/data/rce_cohort_3/*/*.rec/*merged.rec"
all_recording_files = glob.glob(recording_filepath_glob, recursive=True)

all_recording_files = [file_path for file_path in all_recording_files if "copies" not in file_path]

print("Number of recording files: ", len(all_recording_files))

for recording_file in all_recording_files:
    
    print("Current recording file: ", recording_file)
    
    trodes_recording = se.read_spikegadgets(recording_file, stream_id="trodes")       
    trodes_recording = trodes_recording.set_probes(probe_object)
    recording_basename = os.path.basename(recording_file)
    recording_output_directory = "/scratch/back_up/reward_competition_extention/proc/spike_sorting/{}".format(recording_basename)
    
    os.makedirs(recording_output_directory, exist_ok=True)
    print("Output directory: {}".format(recording_output_directory))
    child_spikesorting_output_directory = os.path.join(recording_output_directory,"ss_output")

    try:
        with open('successful_files.txt', "r") as myfile:
            if recording_basename in myfile.read():
                warnings.warn("""Directory already exists for: {}.
                            Either continue on if you are satisfied with the previous run 
                            or delete the directory and run this cell again""".format(recording_basename))
                continue
    except:
        pass

    try:

        start = time.time()
        # Make sure the recording is preprocessed appropriately
        # lazy preprocessing

        recording_filtered = sp.bandpass_filter(trodes_recording, freq_min=300, freq_max=6000, dtype=np.float32)
        recording_preprocessed: si.BaseRecording = sp.whiten(recording_filtered)

        spike_sorted_object = ms5.sorting_scheme2(
            recording=recording_preprocessed,
            sorting_parameters=ms5.Scheme2SortingParameters(
                detect_sign=0,
                phase1_detect_channel_radius=2000,
                detect_channel_radius=2000,
                # other parameters...
                )
                    )

        spike_sorted_object.save(folder=child_spikesorting_output_directory)

        print("Sorting finished in: ", time.time() - start)
                
        sw.plot_rasters(spike_sorted_object)
        plt.title(recording_basename)
        plt.ylabel("Unit IDs")
        
        plt.savefig(os.path.join(recording_output_directory, "{}_raster_plot.png".format(recording_basename)))
        plt.close()
        
        waveform_output_directory = os.path.join(recording_output_directory, "waveforms")
        
        we_spike_sorted = si.extract_waveforms(recording=recording_preprocessed, 
            sorting=spike_sorted_object, folder=waveform_output_directory,
            ms_before=1, ms_after=1, progress_bar=True,
            n_jobs=8, total_memory="16G", overwrite=True,
            max_spikes_per_unit=2000)
        
        phy_output_directory = os.path.join(recording_output_directory, "phy")
        
        export_to_phy(we_spike_sorted, 
                      phy_output_directory,
                      compute_pc_features=True, 
                      compute_amplitudes=True, 
                      remove_if_exists=False)
            
        # edit the params.py file os that it contains the correct realtive path
        params_path = os.path.join(phy_output_directory, "params.py")
        with open(params_path, 'r') as file:
            lines = file.readlines()
        lines[0] = "dat_path = r'./recording.dat'\n"
        with open(params_path, 'w') as file:
            file.writelines(lines)                   

        
    except Exception as e: 
        print(e)
        
        print("Failed sorting: ", recording_basename)
        with open("failed_files.txt", "a") as myfile:
            myfile.write("\n" + recording_basename)

def main():
    """
    Main function that runs when the script is run
    """


if __name__ == '__main__': 
    main()

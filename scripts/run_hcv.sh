#!/bin/bash

# Base paths
DCD_BASE="/home/pwingrp/HCV_CG/HCV_Traj/hcv_trajectory_segments"
PDB_FILE="/home/pwingrp/HCV_CG/HCV_Traj/HCV_110_short.pdb"
GPU_ID=0


for i in $(seq -f "%02g" 1 8)
do
    echo "Processing segment_${i}.dcd..."
    python scripts/parallel_v2.py \
        --dcd_file "${DCD_BASE}/segment_${i}.dcd" \
        --pdb_file "${PDB_FILE}" \
        --gpu ${GPU_ID}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed segment_${i}"
        # Wait for 10 seconds before processing the next segment
        sleep 600
    else
        echo "Error processing segment_${i}"
        exit 1
    fi
    
    echo "----------------------------------------"
done

echo "All segments processed successfully!"

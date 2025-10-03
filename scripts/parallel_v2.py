# This will build an all atom trajectory from a coarse grained Calpha trajectory. It uses the program CGBACK to rebuild the all-atom trajectory.
# It requires a conda environment in which CGBACK is installed.

import os
import timeit
import subprocess
import mdtraj as md
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Rebuild all-atom trajectory from coarse-grained trajectory using CGBACK')
    parser.add_argument('--dcd_file', help='Input DCD trajectory file')
    parser.add_argument('--pdb_file', help='Input PDB topology file')
    parser.add_argument('--gpu', default='0', help='GPU device number to use (default: 0)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Select GPU

start = timeit.default_timer()

# Load coarse grained trajectory.
t = md.load(args.dcd_file, top=args.pdb_file)

t_length = t.n_frames

# clear files in frames and outputs directories
for f in os.listdir('frames'):
   os.remove(os.path.join('frames', f))
for f in os.listdir('outputs'):
   os.remove(os.path.join('outputs', f))

# Save the first frame and generate an all atom model. Save these to serve as reference structures.

t[0].save_pdb('frames/frame0_proc.pdb')

# result = subprocess.run(f"cgback frames/frame0_proc.pdb -o outputs/out0_proc.pdb -d cuda:{args.gpu} --fix-structure-max-iterations 200", shell=True, capture_output=True, text=True)
result = subprocess.run(f"cgback frames/frame0_proc.pdb -o outputs/out0_proc.pdb -d cuda --fix-structure-max-iterations 200", shell=True, capture_output=True, text=True)


# Function to process a single frame
def process_frame(frame_data):
    frame_idx, frame_traj = frame_data
    frame_filename = f'frames/frame{frame_idx}_proc.pdb'
    output_filename = f'outputs/out{frame_idx}_proc.pdb'
    
    # Save frame to PDB
    frame_traj.save_pdb(frame_filename)

    # Run cgback with specific output file
    # result = subprocess.run(f'cgback {frame_filename} -o {output_filename} -d cuda:{args.gpu} --fix-structure-max-iterations 200', 
    #                         shell=True, capture_output=True, text=True)
    result = subprocess.run(f'cgback {frame_filename} -o {output_filename} -d cuda --fix-structure-max-iterations 200', 
                            shell=True, capture_output=True, text=True)

    # Load the result
    frame_result = md.load(output_filename)

    # Clean up files
    if os.path.exists(frame_filename):
        os.remove(frame_filename)
    if os.path.exists(output_filename):
        os.remove(output_filename)

    return (frame_idx, frame_result)

# Create a new trajectory using the all atom model as the first frame.

t_new = md.load('outputs/out0_proc.pdb', top='outputs/out0_proc.pdb')

# Process frames in parallel batches
print(f"Processing {t_length-1} frames in parallel...")

# Determine number of workers (limit to avoid overwhelming GPU)
max_workers = min(multiprocessing.cpu_count(), 4)  # Adjust based on your GPU capacity
# max_workers = 32
print(f"Using {max_workers} parallel workers")

# Process frames in batches of 500
batch_size = 200
total_batches = (t_length + batch_size - 1) // batch_size  # Ceiling division

for batch in range(total_batches):
    start_frame = batch * batch_size + 1
    end_frame = min((batch + 1) * batch_size, t_length)
    print(f"\nProcessing batch {batch + 1}/{total_batches} (frames {start_frame}-{end_frame})...")
    
    # Prepare frame data for this batch
    frame_data = [(i, t[i]) for i in range(start_frame, end_frame)]
    processed_frames = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch jobs
        future_to_frame = {executor.submit(process_frame, data): data[0] for data in frame_data}
        
        # Collect results as they complete
        for future in as_completed(future_to_frame):
            frame_idx = future_to_frame[future]
            result = future.result()    
            processed_frames[result[0]] = result[1]
            print(f"Completed frame {result[0]}")
    
    # Create trajectory for this batch
    if batch == 0:
        batch_traj = t_new
    else:
        # Load previous batch's trajectory
        input_base = Path(args.dcd_file).stem
        prev_start = (batch - 1) * batch_size + 1
        prev_end = min(batch * batch_size, t_length)
        prev_batch_file = f'rebuilt_{input_base}_{prev_start}_{prev_end}.dcd'
        batch_traj = md.load(prev_batch_file, top='outputs/out0_proc.pdb')
    
    # Add processed frames to batch trajectory in order
    print(f"Assembling batch {batch + 1} trajectory...")
    for i in range(start_frame, end_frame):
        if i in processed_frames:
            batch_traj = batch_traj + processed_frames[i]
        else:
            print(f"Warning: Missing frame {i} in batch {batch + 1}")
    
    # Get base name from input DCD file for output naming
    input_base = Path(args.dcd_file).stem
    
    # Save batch trajectory
    batch_filename = f'rebuilt_{input_base}_{start_frame}_{end_frame}.dcd'
    print(f"Saving batch trajectory to {batch_filename}")
    batch_traj.save_dcd(batch_filename)
    
    # Clear processed frames to free memory
    processed_frames.clear()
    del batch_traj

# Final trajectory can be concatenated from batch files if needed

stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+str(execution_time)) # It returns time in seconds        

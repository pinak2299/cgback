# This will build an all atom trajectory from a coarse grained Calpha trajectory. It uses the program CGBACK to rebuild the all-atom trajectory.
# It requires a conda environment in which CGBACK is installed.

import os
import timeit
import subprocess
import mdtraj as md

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start = timeit.default_timer()

# Load coarse grained trajectory.

t = md.load('tests/data/segment_04.dcd', top='tests/data/HCV_110_short.pdb')

t_length = t.n_frames


# Save the first frame and generate an all atom model. Save these to serve as reference structures.
	
t[0].save_pdb('frame0.pdb')
	
subprocess.run('cgback frame0.pdb -d cuda', shell=True)

source = 'out.pdb'
dest = 'out_1.pdb'
os.rename(source, dest)


# Create a new trajectory using the all atom model as the first frame.

t_new = md.load('out_1.pdb', top='out_1.pdb')

#Write out each frame, create an all atom model, and append the aa model to the new trajectory. Delete the 2 pdb files when finished.

for i in range(1,t_length):
	t[i].save_pdb('frame'+str(i)+'.pdb')
	subprocess.run('cgback frame'+str(i)+'.pdb -d cuda --fix-structure-max-iterations 200', shell=True)

	t_add = md.load('out.pdb', top='out_1.pdb')
	t_new = t_new + t_add
	os.remove('out.pdb')
	os.remove('frame'+str(i)+'.pdb')


# Save new all atom trajectory.	

t_new.save_dcd('rebuilt_segment_04.dcd')

stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+str(execution_time)) # It returns time in seconds		

import ray
import time
from msm_we import modelWE
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import pickle
import sys

ray.init()

# Number of MSM microstates to initially put in each stratum/WE bin
# Read in as a parameter
clusters_per_stratum = sys.argv[1]
clusters_per_stratum = int(clusters_per_stratum)

# Following the script shown in msm_we documentation 
# https://msm-we.readthedocs.io/en/latest/_examples/hamsm_construction.html

# WE simulation data
h5file_paths = ['west.h5']

dimreduce_method = 'vamp'

# Boundaries of the basis/target, in progress coordinate space
# 0.103 in basis selects out state 7, which is set to be the only basis state
pcoord_bounds = {
    'basis': [[0, 0.103]],
    'target': [[0.7, 100]]
}

model_name = 'trp-cage unfolding'

# Reference structure
ref_file = '2JOF.pdb'

# WESTPA resampling time
tau = 1e-9

def processCoordinates(self, coords):
    u_ref = mda.Universe(ref_file)
    u_check = mda.Universe(ref_file)
    dist_out = []
    u_check.load_new(coords)
    for frame in u_check.trajectory:
        dists = distances.dist(
            u_check.select_atoms('backbone'),
            u_ref.select_atoms('backbone')
        )[2]
        dist_out.append(dists)
    dist_out = np.array(dist_out)
    return dist_out

modelWE.processCoordinates = processCoordinates

model = modelWE()
model.initialize(
    fileSpecifier=h5file_paths,
    refPDBfile=ref_file,
    modelName=model_name,
    basis_pcoord_bounds=pcoord_bounds['basis'],
    target_pcoord_bounds=pcoord_bounds['target'],
    dim_reduce_method=dimreduce_method,
    tau=tau
)
model.get_iterations()
model.get_coordSet(last_iter = model.maxIter, streaming=True)
model.dimReduce()
model.cluster_coordinates(
    n_clusters=clusters_per_stratum,
    use_ray=True,
    stratified=True,
    store_validation_model=True, # Required for block validation
    random_state=1337
)
model.get_fluxMatrix(n_lag=0)
model.organize_fluxMatrix()
model.get_Tmatrix()
model.get_steady_state()
model.get_committor()
model.get_steady_state_target_flux()
model.do_block_validation(
    cross_validation_groups=2,
    cross_validation_blocks=4
)
model.update_cluster_structures()

# Save the model
with open('model.pkl', 'wb') as of:
    pickle.dump(model, of)

import ray
import time
from msm_we import modelWE
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import pickle
import sys
from msm_we import optimization as mo
from msm_we.stratified_clustering import StratifiedClusters
from westpa.core.binning import RectilinearBinMapper
from msm_we._hamsm._dimensionality import DimensionalityReductionMixin
from sklearn.cluster import KMeans

# version 2
# Reference structure
ref_file = 'reference/2JOF.pdb'

def processCoordinates(coords): #self, coords):
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
    
def CG_matrix(n_cluster, mapping_in_list_form, tmatrix_FG, pi_exact):
    """
    Computes the coarse-grained transition matrix (CG matrix) for a given number of clusters, 
    a mapping of microstates to clusters, and a fine-grained transition matrix.

    Parameters:
        n_cluster (int): The number of clusters.
        mapping_in_list_form (array-like): A 1-dimensional array or list representing the mapping of 
        microstates to clusters.
        tmatrix_FG (array-like): The fine-grained transition matrix.

    Returns:
        array-like: The computed coarse-grained transition matrix (CG matrix).

    Example:
        >>> CG_matrix(3, [0, 1, 0, 2, 1, 2], np.array([[0.2, 0.4, 0.4], [0.6, 0.2, 0.2], [0.3, 0.3, 0.4]]))
        array([[0.45, 0.55, 0.0 ],
               [0.6 , 0.1 , 0.3 ],
               [0.0 , 0.2 , 0.8 ]])
    """
    # Initialize the CG matrix
    tmatrix_CG = np.zeros([n_cluster, n_cluster])
    
    # Convert the mapping to array form
    mapping = np.array(mapping_in_list_form)
    
    # Compute the CG matrix elements
    for m in range(n_cluster):
        for n in range(n_cluster):
            # Get the microstate indices corresponding to clusters m and n
            i_in_m = mapping == m
            j_in_n = mapping == n
            i_in_m = np.where(i_in_m)[0]
            j_in_n = np.where(j_in_n)[0]
        
            # Compute the normalization factor
            norm = np.sum(pi_exact[i_in_m])
            
            # Compute the CG matrix element
            element = 0 
            for i in i_in_m:
                for j in j_in_n:
                    element += pi_exact[i] * tmatrix_FG[i][j]
        
            # Assign the CG matrix element
            tmatrix_CG[m][n] = element / norm
    
    # Normalize the rows of the CG matrix
    for m in range(n_cluster):
        tmatrix_CG[m] /= np.sum(tmatrix_CG[m])
        
    return tmatrix_CG

def partition_in_bin(bin_boundaries, value_list):
    """
    Partition a list of values into bins based on specified bin boundaries.

    This function takes a list of bin boundaries and a list of values and divides
    the values into bins defined by the boundaries. The function returns two
    dictionaries: `bin_vs_list_value` and `bin_vs_list_index`.

    Args:
        bin_boundaries (list): A list of bin boundaries, where each boundary
                              defines the upper limit of a bin. The boundaries
                              should be in ascending order and the list should
                              include the lowest and highest values. No element
                              in the value list should be smaller than the first
                              bin boundary or larger than the last bin boundary.
        value_list (list): A list of numerical values that need to be
                          partitioned into bins.
    Returns:
        tuple: A tuple containing two dictionaries.
            - bin_vs_list_value (dict): A dictionary where the keys are bin
                                        indices and the values are lists of values
                                        that fall within the corresponding bin.
            - bin_vs_list_index (dict): A dictionary where the keys are bin
                                        indices and the values are lists of
                                        indices of elements in the `value_list`
                                        that fall within the corresponding bin.
    Example:
        bin_boundaries = [0, 10, 20, 30]
        value_list = [5, 12, 8, 25, 18]
        bin_vs_list_value, bin_vs_list_index = partition_in_bin(bin_boundaries, value_list)
        # Access values in bin 2
        values_in_bin_2 = bin_vs_list_value[2]
        # Access indices of values in bin 2
        indices_in_bin_2 = bin_vs_list_index[2]
    """
    bin_vs_list_value = {}
    bin_vs_list_index = {}
    for i in range(len(bin_boundaries) - 1):
        indices = np.where((value_list > bin_boundaries[i]) &
                           (value_list <= bin_boundaries[i + 1]))[0]
        bin_vs_list_index[i] = indices
        bin_vs_list_value[i] = value_list[indices]

    return bin_vs_list_value, bin_vs_list_index

def get_exact_mapped_microstate_to_cluster_assignment(pcoord_and_dimreduce, pi_exact, microstate_index_in_bin, cluster_index_in_bin):
    """
    Assign microstates to cluster centers using stratified clustering.

    This function assigns microstates to cluster centers using stratified clustering.
    It takes as input the reduced coordinates (`pcoord_and_dimreduce`), the exact
    stationary distribution (`pi_exact`), indices of microstates in each bin
    (`microstate_index_in_bin`), and indices of cluster centers in each bin
    (`cluster_index_in_bin`).

    Args:
        pcoord_and_dimreduce (numpy.ndarray): Array of reduced coordinates or
                                             dimensionally reduced data of the
                                             microstates.
        pi_exact (numpy.ndarray): Exact stationary distribution for each microstate.
        microstate_index_in_bin (dict): A dictionary where keys are bin indices
                                        and values are lists of indices
                                        corresponding to microstates in that bin.
        cluster_index_in_bin (dict): A dictionary where keys are bin indices
                                     and values are lists of indices corresponding
                                     to cluster centers in that bin.

    Returns:
        tuple: A tuple containing two numpy arrays.
            - microstate_to_cluster_assignment_list (numpy.ndarray): An array
                                                                   where each
                                                                   element
                                                                   represents
                                                                   the assigned
                                                                   cluster center
                                                                   index for a
                                                                   microstate.
            - cluster_centers_exact_mapped (numpy.ndarray): An array containing
                                                           the RMSD values of
                                                           cluster centers for
                                                           the assigned clusters.

    Note:
        The function uses the KMeans clustering algorithm to perform stratified
        clustering on the microstates in each bin. The `pcoord_and_dimreduce`
        array contains the reduced coordinates or dimensionally reduced data of
        microstates. The `pi_exact` array contains the exact stationary distribution
        for each microstate.

    Example:
        pcoord_and_dimreduce = ...  # Reduced coordinates or dimensionally reduced data
        pi_exact = ...              # Exact stationary distribution
        microstate_index_in_bin = {0: [0, 1, 2], 1: [3, 4], ...}  # Indices of microstates in each bin
        cluster_index_in_bin = {0: [0, 1], 1: [2], ...}          # Indices of cluster centers in each bin

        microstate_to_cluster_assignment_list, cluster_centers_exact_mapped = get_exact_mapped_microstate_to_cluster_assignment(
            pcoord_and_dimreduce, pi_exact, microstate_index_in_bin, cluster_index_in_bin
        )
    """
    # Get number of microstates from microstate_index_in_bin
    n_microstate = 0 
    for key, item in microstate_index_in_bin.items():
        n_microstate += len(item)

    # Get number of cluster from cluster_index_in_bin
    n_cluster = 0 
    for key, item in cluster_index_in_bin.items():
        n_cluster += len(item) 
    
    # Microstate to cluster center assignment
    microstate_to_cluster_assignment_list = np.zeros(n_microstate,dtype='int32')
    # RMSD values of cluster centers
    cluster_centers_exact_mapped = np.zeros(n_cluster) 
    
    # Need offset to have distinct cluster indices
    offset = 0 
    
    # Order the bin list to perform stratified clustering on 
    # Basis bin is second to last, target bin is the last 
    n_bin = len(microstate_index_in_bin)
    bin_list = np.array(list(range(1,n_bin-1))+[0,n_bin-1])
    
    # Do 'active' bins first, according to haMSM convention
    for i in bin_list:
        # Values and weights to cluster on 
        X = pcoord_and_dimreduce[microstate_index_in_bin[i]]
        weight = pi_exact[microstate_index_in_bin[i]]
        kmeans = KMeans(n_clusters=len(cluster_index_in_bin[i]),n_init='auto',random_state=53982).fit(X=X,y=None,sample_weight=weight)
    
        # Cluster assignments and RMSD values of cluster centers
        # All the argsort and sort is to make sure that the cluster index is ordered wrt increasing RMSD value 
        microstate_to_cluster_assignment_list[microstate_index_in_bin[i]] = np.argsort(np.argsort(kmeans.cluster_centers_[:,0]))[kmeans.labels_]+offset
        cluster_centers_exact_mapped[np.sort(np.unique(kmeans.labels_+offset))] = np.sort(kmeans.cluster_centers_[:,0])

        '''
        print(f'Bin {i}:')
        print(f'Original label: {kmeans.labels_+offset}')
        print(f'Original RMSD of cluster centers: {kmeans.cluster_centers_[:,0]}')
        print(f'Sorted cluster center RMSD: {np.sort(kmeans.cluster_centers_[:,0])}')
        print(f'New label: {np.argsort(np.argsort(kmeans.cluster_centers_[:,0]))[kmeans.labels_]+offset}')
        print()
        '''
        
        # Adjust offset
        offset += len(np.unique(kmeans.labels_))
    
    return microstate_to_cluster_assignment_list, cluster_centers_exact_mapped

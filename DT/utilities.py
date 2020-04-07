
import numpy as np
import matplotlib.pyplot as plt

from riemann_lab import utils
from riemann_lab import get_datasets as GD
from riemann_lab import diffusion_maps as DM
from riemann_lab import transfer_learning as TL

import moabb
from moabb.paradigms import MotorImagery, P300
from sklearn.externals import joblib

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import powm, invsqrtm
from pyriemann.estimation import Covariances, ERPCovariances

def get_data_epochs(dataset, subject, paradigm=MotorImagery()):

    sessions = dataset._get_single_subject_data(subject)
    sessions_list = list(sessions.keys())
    runs_list = list(sessions[sessions_list[0]].keys())
    raw = sessions[sessions_list[0]][runs_list[0]]
    chnames = raw.pick_types(eeg=True).ch_names
    epochs, labels, _ = paradigm.get_data(dataset, subjects=[subject])
    data = {}
    data['epochs'] = epochs
    data['labels'] = labels
    data['chnames'] = chnames

    return data

# utility functions to augment matrix dimensions
def augment_matrix_dimension(A, ind):
    '''
    - A : input matrix
    - ind : indices of the new matrix where there should be zeros
    '''    

    if len(ind) > 0:

        n = A.shape[0]
        naug = n + len(ind)
        Atilde = np.eye(naug)

        ired = 0
        for iaug in range(naug):
            if iaug not in ind:
                jred = 0
                for jaug in range(naug):
                    if jaug not in ind:
                        Atilde[iaug,jaug] = A[ired,jred]
                        jred = jred + 1
                    else:
                        continue
                ired = ired + 1
            else:
                continue

    else:
        Atilde = A

    return Atilde

def reduce_matrix_dimension(A, ind):

    Atilde = np.delete(A, ind, axis=0)
    Atilde = np.delete(Atilde, ind, axis=1)

    return Atilde    

def augment_dataset_dimension(A, ind):
    '''
    - A : input matrix
    - ind : indices of the new matrix where there should be zeros
    '''
    Atilde = []
    for Ai in A:
        Atilde.append(augment_matrix_dimension(Ai, ind))
    Atilde = np.stack(Atilde)
    return Atilde    

def get_source_target_dataset(source_dataset, source_subject, target_dataset, target_subject):

    # get the data from target dataset
    target = get_data_epochs(target_dataset, target_subject)

    # get the data from source dataset
    source = get_data_epochs(source_dataset, source_subject)

    # get the indices from the expanded matrix
    chnames_total = list(set(source['chnames']).union(set(target['chnames'])))

    idx = {}

    # get the indices for the electrode names on the source dataset
    source_idx_order = []
    source_idx_fill = []
    for i, chi in enumerate(chnames_total):
        if chi in source['chnames']:
            source_idx_order.append(i)
        else:
            source_idx_fill.append(i)
    idx['source_order'] = source_idx_order
    idx['source_fill'] = source_idx_fill

    # get the indices for the electrode names on the target dataset
    target_idx_order = []
    target_idx_fill = []
    for i, chi in enumerate(chnames_total):
        if chi in target['chnames']:
            target_idx_order.append(i)
        else:
            target_idx_fill.append(i)
    idx['target_order'] = target_idx_order
    idx['target_fill'] = target_idx_fill

    return source, target, idx  

def match_source_target_dimensions_motorimagery(source_org, target_org, idx):

    # augment the dimensions for source dataset
    source_org_aug = {}
    dsource = source_org['covs'].shape[1]
    daugment = len(idx['source_fill'])
    idx2fill = np.arange(dsource+daugment)[dsource:]
    source_org_aug['covs'] = augment_dataset_dimension(source_org['covs'], idx2fill) 
    source_org_aug['labels'] = source_org['labels']

    # augment the dimensions for target dataset
    target_org_aug = {}
    dtarget = target_org['covs'].shape[1]
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]    
    target_org_aug['covs'] = augment_dataset_dimension(target_org['covs'], idx2fill) 
    target_org_aug['labels'] = target_org['labels']

    # match the channel orderings for source
    source_org_reo = {}
    idx2order = idx['source_order'] + idx['source_fill']
    source_org_reo['covs'] = source_org_aug['covs'][:,idx2order,:][:,:,idx2order]
    source_org_reo['labels'] = source_org_aug['labels']

    # match the channel orderings for target
    target_org_reo = {}
    idx2order = idx['target_order'] + idx['target_fill']
    target_org_reo['covs'] = target_org_aug['covs'][:,idx2order,:][:,:,idx2order]
    target_org_reo['labels'] = target_org_aug['labels']

    return source_org_reo, target_org_reo  

def match_source_target_dimensions_p300(source_org, target_train_org, target_test_org, idx):

    # augment the dimensions for source dataset
    source_org_aug = {}
    dsource = len(idx['source_order'])
    daugment = len(idx['source_fill'])
    idx2fill = np.arange(dsource+daugment)[dsource:]
    idx2fill = np.concatenate([idx2fill, (dsource+daugment)+idx2fill])
    source_org_aug['covs'] = augment_dataset_dimension(source_org['covs'], idx2fill) 
    source_org_aug['labels'] = source_org['labels']

    # augment the dimensions for target training dataset
    target_train_org_aug = {}
    dtarget = len(idx['target_order'])
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]
    idx2fill = np.concatenate([idx2fill, (dtarget+daugment)+idx2fill])
    target_train_org_aug['covs'] = augment_dataset_dimension(target_train_org['covs'], idx2fill) 
    target_train_org_aug['labels'] = target_train_org['labels']

    # augment the dimensions for target dataset
    target_test_org_aug = {}
    dtarget = len(idx['target_order'])
    daugment = len(idx['target_fill'])
    idx2fill = np.arange(dtarget+daugment)[dtarget:]
    idx2fill = np.concatenate([idx2fill, (dtarget+daugment)+idx2fill])
    target_test_org_aug['covs'] = augment_dataset_dimension(target_test_org['covs'], idx2fill) 
    target_test_org_aug['labels'] = target_test_org['labels']    

    # match the channel orderings for source
    source_org_reo = {}
    idx2order = np.array(idx['source_order'] + idx['source_fill'])
    idx2order = np.concatenate([idx2order, len(idx2order)+idx2order])
    source_org_reo['covs'] = source_org_aug['covs'][:,idx2order,:][:,:,idx2order]
    source_org_reo['labels'] = source_org_aug['labels']

    # match the channel orderings for target
    target_train_org_reo = {}
    idx2order = np.array(idx['target_order'] + idx['target_fill'])
    idx2order = np.concatenate([idx2order, len(idx2order)+idx2order])
    target_train_org_reo['covs'] = target_train_org_aug['covs'][:,idx2order,:][:,:,idx2order]
    target_train_org_reo['labels'] = target_train_org_aug['labels']

    # match the channel orderings for target
    target_test_org_reo = {}
    idx2order = np.array(idx['target_order'] + idx['target_fill'])
    idx2order = np.concatenate([idx2order, len(idx2order)+idx2order])
    target_test_org_reo['covs'] = target_test_org_aug['covs'][:,idx2order,:][:,:,idx2order]
    target_test_org_reo['labels'] = target_test_org_aug['labels']    

    return source_org_reo, target_train_org_reo, target_test_org_reo 

def parallel_transport_covariance_matrix(C, R):
    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):
    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)     
    
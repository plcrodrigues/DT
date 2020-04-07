
import joblib
import numpy as np

from riemann_lab import transfer_learning as TL
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

from tqdm import tqdm
from DT import utilities

from moabb.datasets import BNCI2015001, AlexMI
from moabb.paradigms import MotorImagery

def scores_transfer_learning_cross_validation(clf, source, target, ncovs_train, nrzt=5):

    scores_subj = {}
    scr_ntr_rzt = []
    scr_org_rzt = []
    scr_rct_rzt = []
    scr_str_rzt = []    
    scr_rot_rzt = []

    for _ in range(nrzt):

        # split target into training and testing for RPA
        target_train = {}
        target_test = {}
        source['org-aug'], target_train['org-aug'], target_test['org-aug'] = TL.get_sourcetarget_split(source['org-aug'], target['org-aug'], ncovs_train)
        # match geometric means (RPA recentering)
        source['rct-aug'], target_train['rct-aug'], target_test['rct-aug'] = TL.RPA_recenter(source['org-aug'], target_train['org-aug'], target_test['org-aug'])
        # match dispersions (RPA stretching)
        source['str-aug'], target_train['str-aug'], target_test['str-aug'] = TL.RPA_stretch(source['rct-aug'], target_train['rct-aug'], target_test['rct-aug'])
        # match class means (RPA rotation)
        source['rot-aug'], target_train['rot-aug'], target_test['rot-aug'] = TL.RPA_rotate(source['str-aug'], target_train['str-aug'], target_test['str-aug'])

        # get the scores
        scr_ntr_rzt = scr_ntr_rzt + [TL.get_score_notransfer(clf, target_train['org-aug'], target_test['org-aug'])]
        scr_org_rzt = scr_org_rzt + [TL.get_score_transferlearning(clf, source['org-aug'], target_train['org-aug'], target_test['org-aug'])]
        scr_rct_rzt = scr_rct_rzt + [TL.get_score_transferlearning(clf, source['rct-aug'], target_train['rct-aug'], target_test['rct-aug'])]
        scr_str_rzt = scr_str_rzt + [TL.get_score_transferlearning(clf, source['str-aug'], target_train['str-aug'], target_test['str-aug'])]
        scr_rot_rzt = scr_rot_rzt + [TL.get_score_transferlearning(clf, source['rot-aug'], target_train['rot-aug'], target_test['rot-aug'])]

    scores_subj['ntr'] = (np.mean(scr_ntr_rzt), np.var(scr_ntr_rzt))
    scores_subj['org'] = (np.mean(scr_org_rzt), np.var(scr_org_rzt))
    scores_subj['rct'] = (np.mean(scr_rct_rzt), np.var(scr_rct_rzt))
    scores_subj['str'] = (np.mean(scr_str_rzt), np.var(scr_str_rzt))  
    scores_subj['rot'] = (np.mean(scr_rot_rzt), np.var(scr_rot_rzt))

    return scores_subj

def get_results(clf, source_dataset, source_subject, target_dataset, target_subject, ncovs_target_train_list):

    scores_target = {}

    print('target subject:', target_subject, ', source subject:', source_subject)

    # get the data from source dataset
    dataset_source, dataset_target, idx = utilities.get_source_target_dataset(source_dataset, source_subject, target_dataset, target_subject)

    # estimate the covariances
    source = {}
    source['org'] = {}
    source['org']['covs'] = Covariances(estimator='lwf').fit_transform(dataset_source['epochs'])
    source['org']['labels'] = dataset_source['labels']
    target = {}
    target['org'] = {}
    target['org']['covs'] = Covariances(estimator='lwf').fit_transform(dataset_target['epochs'])
    target['org']['labels'] = dataset_target['labels']

    # match the dimensions (reorder and expand)
    source['org-aug'], target['org-aug'] = utilities.match_source_target_dimensions_motorimagery(source['org'], target['org'], idx)

    # get the scores
    scores_target[source_subject] = {}
    for ncovs_target_train in ncovs_target_train_list:
        scores_target[source_subject][ncovs_target_train] = scores_transfer_learning_cross_validation(clf, source, target, ncovs_target_train, nrzt=5)

    return scores_target

# choose the target dataset and which subject to use
target_dataset = AlexMI()
target_subject = 2

# choose the source dataset and which subject to use
source_dataset = BNCI2015001()
source_subject = 1

# obtain the cross-subject classification scores
clf = MDM()       
ncovs_target_train_list = [1, 5, 10] 
scores_target = get_results(clf, source_dataset, source_subject, target_dataset, target_subject, ncovs_target_train_list)
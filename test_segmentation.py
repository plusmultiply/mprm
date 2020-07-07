

# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.tester import ModelTester
from models.KPCNN_model import KernelPointCNN
from models.KPFCNN_attention_model import KernelPointFCNN

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset
from datasets.S3DIS import S3DISDataset
from datasets.Scannet_attention import ScannetDataset
from datasets.NPM3D import NPM3DDataset
from datasets.Semantic3D import Semantic3DDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def test_caller(path, step_ind, on_val):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '3'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_color = 1.0
    config.validation_size = 500
    #config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration

    dataset = ScannetDataset(config.input_threads, load_test=(not on_val))


    # Create subsample clouds of the models
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initialize input pipelines
    if on_val:
        dataset.init_input_pipeline(config)
    else:
        dataset.init_test_input_pipeline(config)


    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()


    model = KernelPointFCNN(dataset.flat_inputs, config)


    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    if config.dataset.startswith('ShapeNetPart'):
        if config.dataset.split('_')[1] == 'multi':
            tester.test_multi_segmentation(model, dataset)
        else:
            tester.test_segmentation(model, dataset)
    elif config.dataset.startswith('S3DIS'):
        tester.test_cloud_segmentation_on_val(model, dataset)
    elif config.dataset.startswith('Scannet'):
        if on_val:
            tester.test_cloud_segmentation_on_val(model, dataset)
        else:
            tester.test_cloud_segmentation(model, dataset)
    elif config.dataset.startswith('Semantic3D'):
        if on_val:
            tester.test_cloud_segmentation_on_val(model, dataset)
        else:
            tester.test_cloud_segmentation(model, dataset)
    elif config.dataset.startswith('NPM3D'):
        if on_val:
            tester.test_cloud_segmentation_on_val(model, dataset)
        else:
            tester.test_cloud_segmentation(model, dataset)
    elif config.dataset.startswith('ModelNet40'):
        tester.test_classification(model, dataset)
    else:
        raise ValueError('Unsupported dataset')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Choose the model to test
    ##########################

    
    #specify your log path
    chosen_log = ''

    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = -1

    #
    #   Eventually, you can choose to test your model on the validation set
    #

    on_val = True
    

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot, on_val)




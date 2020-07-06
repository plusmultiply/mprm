#      Modified by Wei Jiacheng
#
#
#      Originally written by Hugues THOMAS - 11/06/2018
#




# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix

from tensorflow.python.client import timeline
import json


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits_softmax = tf.nn.softmax(model.logits)
        self.prob_logits = model.logits

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_classification(self, model, dataset, num_votes=30):

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate votes
        average_probs = np.zeros((len(dataset.input_labels['test']), nc_model))
        average_counts = np.zeros((len(dataset.input_labels['test']), nc_model))

        mean_dt = np.zeros(2)
        last_display = time.time()
        while np.min(average_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []
            count = 0

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits, model.labels, model.inputs['object_inds'])
                    prob, labels, inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Get probs and labels
                    probs += [prob]
                    targets += [labels]
                    obj_inds += [inds]
                    count += prob.shape[0]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(np.min(average_counts),
                                             100 * count / dataset.num_test,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Average votes
            # *************

            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(dataset.input_labels['test'][obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered)
            average_counts[obj_inds] += 1
            average_probs[obj_inds] += (probs - average_probs[obj_inds]) / (average_counts[obj_inds])

            # Save/Display temporary results
            # ******************************

            test_labels = np.array(dataset.label_values)

            # Compute classification results
            C1 = confusion_matrix(dataset.input_labels['test'],
                                  np.argmax(average_probs, axis=1),
                                  test_labels)

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print('Test Accuracy = {:.1f}%'.format(ACC))

            s = ''
            for cc in C1:
                for c in cc:
                    s += '{:d} '.format(c)
                s += '\n'
            print(s)



            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return



    def test_cloud_segmentation(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.data.shape[0], nc_model), dtype=np.float32) for l in dataset.input_trees['test']]

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops,
                                                                                       {model.dropout_prob: 1.0})
                """
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops,
                                                                                       {model.dropout_prob: 1.0},
                                                                                       options=options,
                                                                                       run_metadata=run_metadata)
                """
                t += [time.time()]

                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #many_runs_timeline.update_timeline(chrome_trace)

                if False:
                    many_runs_timeline.save('timeline_merged_%d_runs.json' % i0)
                    a = 1/0

                # Get predictions and labels per instance
                # ***************************************

                # Stack all predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs



                # Average timing
                t += [time.time()]
                #print(batches.shape, stacked_probs.shape, 1000*(t[1] - t[0]), 1000*(t[2] - t[1]))
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['test'])))

                i0 += 1

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['test'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))
                print([np.mean(pots) for pots in dataset.potentials['test']])

                if last_min + 2 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):

                        # Get file
                        points = dataset.load_evaluation_points(file_path)

                        # Reproject probs
                        probs = self.test_probs[i_test][dataset.test_proj[i_test], :]

                        # Insert false columns for ignored labels
                        probs2 = probs.copy()
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.int32)

                        # Project potentials on original points
                        pots = dataset.potentials['test'][i_test][dataset.test_proj[i_test]]

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds, pots],
                                  ['x', 'y', 'z', 'preds', 'pots'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values
                                      if label not in dataset.ignored_labels]
                        write_ply(test_name2,
                                  [points, probs],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save ascii preds
                        if dataset.name.startswith('Semantic3D'):
                            ascii_name = join(test_path, 'predictions', dataset.ascii_files[cloud_name])
                        else:
                            ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(dataset.test_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return

    def test_cloud_segmentation_on_val(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]
        self.test_probs_softmax = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions')):
                makedirs(join(test_path, 'val_predictions'))
            if not exists(join(test_path, 'val_probs')):
                makedirs(join(test_path, 'val_probs'))
            if not exists(join(test_path, 'val_probs_softmax')):
                makedirs(join(test_path, 'val_probs_softmax'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       self.prob_logits_softmax,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, softmax_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    sm_probs = softmax_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs
                    self.test_probs_softmax[c_i][inds] = test_smooth * self.test_probs_softmax[c_i][inds] + (1 - test_smooth) * sm_probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))

                i0 += 1


            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i_test in range(dataset.num_validation):

                        # Insert false columns for ignored labels
                        probs = self.test_probs_softmax[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = dataset.input_labels['validation'][i_test]

                        # Confs
                        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                    if int(np.ceil(new_min)) % 2 == 0:

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        t1 = time.time()
                        files = dataset.train_files
                        i_val = 0
                        proj_probs = []
                        proj_probs_softmax = []
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == 0:

                                # Reproject probs on the evaluations points
                                probs = self.test_probs[i_val][dataset.validation_proj[i_val], :]
                                proj_probs += [probs]
                                probs_sm = self.test_probs_softmax[i_val][dataset.validation_proj[i_val], :]
                                proj_probs_softmax += [probs_sm]
                                i_val += 1

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i_test in range(dataset.num_training):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    proj_probs_softmax[i_test] = np.insert(proj_probs_softmax[i_test], l_ind, 0, axis=1)
                                    proj_probs[i_test] = np.insert(proj_probs[i_test], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs_softmax[i_test], axis=1)].astype(np.int32)

                            # Confusion
                            targets = dataset.validation_labels[i_test]
                            Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        files = dataset.train_files
                        i_test = 0
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.training_split:
                                cloud_name = file_path.split('/')[-1][:-4]

                                # Get points
                                points, rgb = dataset.load_evaluation_points_on_train(cloud_name)

                                # Get the predicted labels
                                preds = dataset.label_values[np.argmax(proj_probs_softmax[i_test], axis=1)].astype(np.int32)

                                # Project potentials on original points
                                pots = dataset.potentials['validation'][i_test][dataset.validation_proj[i_test]]

                                # Save plys
                                #cloud_name = file_path.split('/')[-1]
                                #test_name = join(test_path, 'val_predictions', cloud_name)
                                #write_ply(test_name,
                                #          [points, preds, pots, dataset.validation_labels[i_test]],
                                #          ['x', 'y', 'z', 'preds', 'pots', 'gt'])
                                test_name2 = join(test_path, 'val_probs', cloud_name)
                                prob_names = ['_'.join(dataset.label_to_names[label].split())
                                              for label in dataset.label_values]
                                write_ply(test_name2,
                                          [points, rgb, dataset.validation_labels[i_test], proj_probs[i_test]],
                                          ['x', 'y', 'z', 'red', 'green', 'blue', 'gt'] + prob_names)
                                #test_name3 = join(test_path, 'val_probs_softmax', cloud_name)
                                #write_ply(test_name3,
                                #          [points, rgb, dataset.validation_labels[i_test], proj_probs_softmax[i_test]],
                                #          ['x', 'y', 'z', 'red', 'green', 'blue', 'gt'] + prob_names)
                                i_test += 1
                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))


                self.sess.run(dataset.val_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return































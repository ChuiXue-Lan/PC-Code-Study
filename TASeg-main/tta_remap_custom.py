#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import multiprocessing
from pathlib import Path

# possible splits
splits = ["train", "valid", "trainval", "test"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./remap_semantic_labels.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default=None,
        help='Dataset dir. WARNING: This file remaps the labels in place, so the original labels will be lost. Cannot be used together with -predictions- flag.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=False,
        default=None,
        help='Output directory for remapped labels. If not specified, will create dataset_remap directory.'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default="valid",
        help='Split to evaluate on. One of ' +
        str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--datacfg', '-dc',
        type=str,
        required=False,
        default="semantic-kitti-all.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--inverse',
        dest='inverse',
        default=False,
        action='store_true',
        help='Map from xentropy to original, instead of original to xentropy. '
        'Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Output: ", FLAGS.output)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.datacfg)
    print("Inverse: ", FLAGS.inverse)
    print("*" * 80)

    # check dataset path
    assert FLAGS.dataset is not None, "Dataset path must be provided!"
    
    # setup output directory
    if FLAGS.output is None:
        FLAGS.output = FLAGS.dataset + "_remap"
    
    # assert split
    assert(FLAGS.split in splits)

    print("Opening data config file %s" % FLAGS.datacfg)
    DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

    # get number of interest classes, and the label mappings
    if FLAGS.inverse:
        print("Mapping xentropy to original labels")
        remapdict = DATA["learning_map_inv"]
    else:
        remapdict = DATA["learning_map"]
    nr_classes = len(remapdict)

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # get wanted set
    sequences = []
    sequences.extend(DATA["split"][FLAGS.split])

    def remap_single_sequence(sequence):
        sequence = '{0:02d}'.format(int(sequence))
        print(f"Processing sequence {sequence}")
        
        # Setup paths
        seq_path = os.path.join(FLAGS.dataset, sequence)
        seq_path_out = os.path.join(FLAGS.output, sequence)
        label_dir = os.path.join(seq_path, "labels")
        label_dir_out = os.path.join(seq_path_out, "labels")
        
        # Create output directories
        os.makedirs(seq_path_out, exist_ok=True)
        os.makedirs(label_dir_out, exist_ok=True)
        
        # Copy velodyne and calib directories
        os.system(f"cp -r {os.path.join(seq_path, 'velodyne')} {seq_path_out}/")
        os.system(f"cp -r {os.path.join(seq_path, 'calib')} {seq_path_out}/")
        
        # Get all label files
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.label')]
        label_files.sort()
        
        for label_file in label_files:
            input_path = os.path.join(label_dir, label_file)
            output_path = os.path.join(label_dir_out, label_file)
            
            print(f"Remapping {input_path}")
            label = np.fromfile(input_path, dtype=np.uint32)
            label = label.reshape((-1))
            upper_half = label >> 16      # get upper half for instances
            lower_half = label & 0xFFFF   # get lower half for semantics
            lower_half = remap_lut[lower_half]  # do the remapping of semantics
            label = (upper_half << 16) + lower_half   # reconstruct full label
            label = label.astype(np.uint32)
            label.tofile(output_path)

    # Process all sequences
    with multiprocessing.Pool(processes=min(8, len(sequences))) as pool:
        pool.map(remap_single_sequence, sequences)

    print("Done remapping labels!") 
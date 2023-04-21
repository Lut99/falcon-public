#!/usr/bin/env python3
# GENERATE CIFAR10.py
#   by Lut99
#
# Created:
#   22 Dec 2022, 14:23:55
# Last edited:
#   25 Jan 2023, 09:52:02
# Auto updated?
#   Yes
#
# Description:
#   Generates CIFAR10 data files by snatching it from the `keras` library.
#

import argparse
import os
import sys

from keras.datasets import cifar10
import numpy as np
#from tensorflow.image import resize


##### MAIN #####
def main(output_dir: str, homogeneous: bool) -> int:
    # Load the CIFAR10 dataset
    (train_imgs, train_lbls), (test_imgs, test_lbls) = cifar10.load_data()

    # Resize the images
    #print("Resizing images from 32x32 to 33x33 (damn you Falcon)...")
    #train_imgs = np.array([ resize(img, (33, 33)) for img in train_imgs ], dtype=np.float64)
    ##test_imgs  = np.array([ resize(img, (33, 33)) for img in test_imgs ], dtype=np.float64)

    # Split it into three parts for three parties
    if not homogeneous:
        train_imgs = {
            "A" : train_imgs[     :16666],
            "B" : train_imgs[16666:33334],
            "C" : train_imgs[33334:],
        }
        train_lbls = {
            "A" : train_lbls[     :16666],
            "B" : train_lbls[16666:33334],
            "C" : train_lbls[33334:],
        }

        test_imgs = {
            "A" : test_imgs[    :3333],
            "B" : test_imgs[3333:6667],
            "C" : test_imgs[6667:],
        }
        test_lbls = {
            "A" : test_lbls[    :3333],
            "B" : test_lbls[3333:6667],
            "C" : test_lbls[6667:],
        }
    else:
        train_imgs = {
            "A" : train_imgs,
            "B" : train_imgs,
            "C" : train_imgs,
        }
        train_lbls = {
            "A" : train_lbls,
            "B" : train_lbls,
            "C" : train_lbls,
        }

        test_imgs = {
            "A" : test_imgs,
            "B" : test_imgs,
            "C" : test_imgs,
        }
        test_lbls = {
            "A" : test_lbls,
            "B" : test_lbls,
            "C" : test_lbls,
        }

    # Write them to separate files
    for party in [ "A", "B", "C" ]:
        # Loop to write a test and training set
        for (kind, data, labels) in [ ("train", train_imgs[party], train_lbls[party]), ("test", test_imgs[party], test_lbls[party]) ]:
            # Write the dataset
            path = os.path.join(output_dir, f"{kind}_data_{party}")
            print(f"Generating '{path}' ({data.shape[0]} samples, {data.shape[1]}x{data.shape[2]} images)")
            if party == "A":
                try:
                    with open(path, "w") as h:
                        # Generate one image per sample
                        for i in range(data.shape[0]):
                            # Generate width * height pixels...
                            for y in range(data.shape[1]):
                                for x in range(data.shape[2]):
                                    # ...with pixel pixels each
                                    for p in range(data.shape[3]):
                                        h.write(f"{data[i, y, x, p]} ")
                except IOError as e:
                    print(f"ERROR: Failed to write to '{path}': {e}", file=sys.stderr)
                    return 1

                # Generate the file with the labels
                path = os.path.join(output_dir, f"{kind}_labels_{party}")
                print(f"Generating '{path}' ({labels.shape[0]} samples, 10 output classes per sample)")
                try:
                    with open(path, "w") as h:
                        # Generate one string of floats per sample
                        for i in range(labels.shape[0]):
                            # Generate last_layer value
                            for j in range(10):
                                h.write(f"{1 if labels[i] == j else 0} ")
                except IOError as e:
                    print(f"ERROR: Failed to write to '{path}': {e}", file=sys.stderr)
                    return 1
            else:
                try:
                    with open(path, "w") as h:
                        # Generate one image per sample
                        for i in range(data.shape[0]):
                            # Generate width * height pixels...
                            for y in range(data.shape[1]):
                                for x in range(data.shape[2]):
                                    # ...with pixel pixels each
                                    for p in range(data.shape[3]):
                                        h.write(f"{0} ")
                except IOError as e:
                    print(f"ERROR: Failed to write to '{path}': {e}", file=sys.stderr)
                    return 1

                # Generate the file with the labels
                path = os.path.join(output_dir, f"{kind}_labels_{party}")
                print(f"Generating '{path}' ({labels.shape[0]} samples, 10 output classes per sample)")
                try:
                    with open(path, "w") as h:
                        # Generate one string of floats per sample
                        for i in range(labels.shape[0]):
                            # Generate last_layer value
                            for j in range(10):
                                h.write(f"{0 if labels[i] == j else 0} ")
                except IOError as e:
                    print(f"ERROR: Failed to write to '{path}': {e}", file=sys.stderr)
                    return 1
    # Done
    return 0



# Actual entrypoint to the file
if __name__ == "__main__":
    # Define the arguments to parse
    parser = argparse.ArgumentParser()
    parser.add_argument("OUTPUT_DIR", default="./files", help="The output directory to generate the files in. Will complain if it doesn't exist yet.")
    parser.add_argument("-H", "--homogeneous", action="store_true", help="If given, does not split the dataset three times but rather copies it three times for every party.")

    # Parse 'em
    args = parser.parse_args()

    # Run the main function
    exit(main(args.OUTPUT_DIR, args.homogeneous))

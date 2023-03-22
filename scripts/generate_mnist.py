#!/usr/bin/env python3
# GENERATE MNIST.py
#   by Lut99
#
# Created:
#   22 Dec 2022, 14:23:55
# Last edited:
#   22 Mar 2023, 20:02:27
# Auto updated?
#   Yes
#
# Description:
#   Generates MNIST data files by snatching it from the `keras` library.
#

import argparse
import os
import sys

from keras.datasets import mnist


##### MAIN #####
def main(output_dir: str, mode: str) -> int:
    # Load the MNIST datraset
    (train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()

    # Split it into three parts for three parties
    if mode == "split":
        train_imgs = {
            "A" : train_imgs[     :20000],
            "B" : train_imgs[20000:40000],
            "C" : train_imgs[40000:],
        }
        train_lbls = {
            "A" : train_lbls[     :20000],
            "B" : train_lbls[20000:40000],
            "C" : train_lbls[40000:],
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
    elif mode == "duplicate":
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
    elif mode == "secret_share":
        # This is gon' be interesting
        pass

    # Write them to separate files
    for party in [ "A", "B", "C" ]:
        # Loop to write a test and training set
        for (kind, data, labels) in [ ("train", train_imgs[party], train_lbls[party]), ("test", test_imgs[party], test_lbls[party]) ]:
            # Write the dataset
            path = os.path.join(output_dir, f"{kind}_data_{party}")
            print(f"Generating '{path}' ({data.shape[0]} samples, {data.shape[1]}x{data.shape[2]} images)")
            try:
                with open(path, "w") as h:
                    # Generate one image per sample
                    for i in range(data.shape[0]):
                        # Generate width * height pixels...
                        for y in range(data.shape[1]):
                            # ...with pixel pixels each
                            for x in range(data.shape[2]):
                                h.write(f"{data[i, y, x]} ")
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

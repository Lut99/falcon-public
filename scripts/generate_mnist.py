#!/usr/bin/env python3
# GENERATE MNIST.py
#   by Lut99
#
# Created:
#   22 Dec 2022, 14:23:55
# Last edited:
#   27 Mar 2023, 18:34:06
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
    
    parties = []
    if mode == "split":
        parties = [ "A", "B", "C" ]
        train_imgs = {
            "_A" : train_imgs[     :20000],
            "_B" : train_imgs[20000:40000],
            "_C" : train_imgs[40000:],
        }
        train_lbls = {
            "_A" : train_lbls[     :20000],
            "_B" : train_lbls[20000:40000],
            "_C" : train_lbls[40000:],
        }

        test_imgs = {
            "_A" : test_imgs[    :3333],
            "_B" : test_imgs[3333:6667],
            "_C" : test_imgs[6667:],
        }
        test_lbls = {
            "_A" : test_lbls[    :3333],
            "_B" : test_lbls[3333:6667],
            "_C" : test_lbls[6667:],
        }
    elif mode == "duplicate":
        parties = [ "A", "B", "C" ]
        train_imgs = {
            "_A" : train_imgs,
            "_B" : train_imgs,
            "_C" : train_imgs,
        }
        train_lbls = {
            "_A" : train_lbls,
            "_B" : train_lbls,
            "_C" : train_lbls,
        }

        test_imgs = {
            "_A" : test_imgs,
            "_B" : test_imgs,
            "_C" : test_imgs,
        }
        test_lbls = {
            "_A" : test_lbls,
            "_B" : test_lbls,
            "_C" : test_lbls,
        }
    elif mode == "no_share":
        parties = [ "" ]
        train_imgs = {
            "" : train_imgs,
        }
        train_lbls = {
            "" : train_lbls,
        }

        test_imgs = {
            "" : test_imgs,
        }
        test_lbls = {
            "" : test_lbls,
        }

    # Write them to separate files
    for party in parties:
        # Loop to write a test and training set
        for (kind, data, labels) in [ ("train", train_imgs[party], train_lbls[party]), ("test", test_imgs[party], test_lbls[party]) ]:
            # Write the dataset
            path = os.path.join(output_dir, f"{kind}_data{party}")
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
            path = os.path.join(output_dir, f"{kind}_labels{party}")
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
    parser.add_argument("-m", "--mode", default="split", help="The mode that we use to compute the split in various parties. Can be 'split', 'duplicate' or 'secret_share'.")

    # Parse 'em
    args = parser.parse_args()

    # Run the main function
    exit(main(args.OUTPUT_DIR, args.mode))

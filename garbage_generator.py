#!/usr/bin/env python3
# GARBAGE GENERATOR.py
#   by Tim Müller
# 
# Quick script that generates the twelve dataset files in a way that causes training of falcon to not crash.
# 
# However, note that this is by no means real data. Please download the actual MNIST, CIFAR10 or ImageNet datasets for that.
# 

import argparse
import os
import random
import sys


##### MAIN #####
def main(output_dir, train_N, test_N, width, height, pixel, last_layer):
    # Loop to generate the files for each party
    for party in [ "A", "B", "C" ]:
        # Loop to generate a test and training set
        for (kind, n_samples) in [ ("train", train_N), ("test", test_N) ]:
            # Generate the file with the data
            path = os.path.join(output_dir, f"{kind}_data_{party}")
            print(f"Generating '{path}' ({n_samples} samples, {width}x{height} images, {pixel} colours per pixel)")
            try:
                with open(path, "w") as h:
                    # Generate one image per sample
                    for _ in range(n_samples):
                        # Generate width * height pixels...
                        for _ in range(width * height):
                            # ...with pixel pixels each
                            for _ in range(pixel):
                                h.write(f"{random.random()} ")
            except IOError as e:
                print(f"ERROR: Failed to write to '{path}': {e}", file=sys.stderr)
                return 1

            # Generate the file with the labels
            path = os.path.join(output_dir, f"{kind}_labels_{party}")
            print(f"Generating '{path}' ({n_samples} samples, {last_layer} output values per sample)")
            try:
                with open(path, "w") as h:
                    # Generate one string of floats per sample
                    for _ in range(n_samples):
                        # Generate last_layer value
                        for _ in range(last_layer):
                            h.write(f"{random.random()} ")
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
    parser.add_argument("-n", "--train-N", type=int, default="8", help="The number of samples to generate in each train dataset.")
    parser.add_argument("-N", "--test-N", type=int, default="8", help="The number of samples to generate in each test dataset.")
    parser.add_argument("-W", "--width", type=int, help="The width of the images to generate.")
    parser.add_argument("-H", "--height", type=int, help="The height of the images to generate.")
    parser.add_argument("-P", "--pixel", type=int, help="The number of values within each pixel.")
    parser.add_argument("-L", "--last-layer", type=int, help="The number of nodes in the algorithm's last layer.")

    parser.add_argument("--mnist", action="store_true", help="Use the default values for MNIST, as given in the default secondary.cpp")
    parser.add_argument("--cifar10", action="store_true", help="Use the default values for CIFAR10, as given in the default secondary.cpp (we assume AlexNet)")
    parser.add_argument("--image-net", action="store_true", help="Use the default values for ImageNet, as given in the default secondary.cpp (we assume AlexNet)")

    # Parse 'em
    args = parser.parse_args()

    # Fill in some defaults
    if args.mnist:
        args.width      = 28
        args.height     = 28
        args.pixel      = 1
        args.last_layer = 10
    if args.cifar10:
        args.width      = 33
        args.height     = 33
        args.pixel      = 3
        args.last_layer = 10
    if args.image_net:
        args.width      = 56
        args.height     = 56
        args.pixel      = 3
        args.last_layer = 200
    if not args.mnist and not args.cifar10 and not args.image_net:
        if args.width is None:
            print("Specify a value for '--width' (or use '--mnist', '--cifar10' or '--image-net'", file=sys.stderr)
            exit(1)
        if args.height is None:
            print("Specify a value for '--height' (or use '--mnist', '--cifar10' or '--image-net'", file=sys.stderr)
            exit(1)
        if args.pixel is None:
            print("Specify a value for '--pixel' (or use '--mnist', '--cifar10' or '--image-net'", file=sys.stderr)
            exit(1)
        if args.last_layer is None:
            print("Specify a value for '--last-layer' (or use '--mnist', '--cifar10' or '--image-net'", file=sys.stderr)
            exit(1)

    # Run the main function
    exit(main(args.OUTPUT_DIR, args.train_N, args.test_N, args.width, args.height, args.pixel, args.last_layer))

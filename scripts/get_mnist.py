#!/usr/bin/env python3
# GET MNIST.py
#   by Lut99
#
# Created:
#   16 Jan 2023, 14:10:56
# Last edited:
#   16 Jan 2023, 15:04:12
# Auto updated?
#   Yes
#
# Description:
#   Script that pulls the MNIST dataset from the interwebs in a format that
#   falcon understands.
#

import argparse
import gzip
import os
import random
import urllib.error
import urllib.request
import urllib.response
import string
import sys
import time


##### HELPER FUNCTIONS #####
def bytes_to_string(b: int) -> str:
    """
        Creates a human-friendly representation of the given amount of bytes.
    """

    if b < 1_000: return f"{b}B"
    if b < 1_000_000: return f"{b / 1_000.0:.2f}KB"
    if b < 1_000_000_000: return f"{b / 1_000_000.0:.2f}MB"
    return f"{b / 1_000_000_000.0:.2f}GB"

def download(url: str, target: str) -> int:
    """
        Will attempt to download the file at the given URL to the given path.

        Uses an HTTP request to do so.

        # Arguments
        - `url`: The URL to download from.
        - `target`: The target path to download to.

        # Returns
        The return status of the process. `0` means success.
    """

    # Open the connection
    try:
        req = urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        print(f"Failed to open request to '{url}': {e}", file=sys.stderr)
        return e.errno

    # Stream the result to the file
    try:
        with open(target, "wb") as h:
            done  = 0
            total = int(req.headers["Content-Length"]) if req.headers["Content-Length"] is not None else None
            speed = 0
            print(f"Downloading '{url}' {bytes_to_string(done)}/{bytes_to_string(total)}... (0B/s)", end='\r')
            last_update = time.time()

            chunk = req.read(65535)
            while len(chunk) > 0:
                # Write the chunk
                elapsed = time.time() - last_update
                if elapsed >= 0.25:
                    print(f"Downloading '{url}' {bytes_to_string(done)}/{bytes_to_string(total)}... ({bytes_to_string((1.0 / elapsed) * speed)}/s)", end='\r')
                    last_update = time.time()
                    speed = 0
                h.write(chunk)
                done  += len(chunk)
                speed += len(chunk)

                # Get the next chunk
                chunk = req.read(65535)
            print(f"Downloading '{url}' {bytes_to_string(done)}/{bytes_to_string(total)}... ({bytes_to_string((1.0 / (time.time() - last_update)) * speed)}/s)")

    except IOError as e:
        print(f"\nFailed to write chunk to '{target}': {e}", file=sys.stderr)
        return e.errno

    # Done
    return 0

def unarchive(source: str, target: str) -> int:
    """
        Unarchives the given `source` archive to the given `target` directory.

        You should create a directory yourself if you don't want to clutter `target`'s root.

        # Arguments
        - `source`: The path to the archive to get rid of.
        - `target`: The path to the directory where to copy the contents to.

        # Returns
        The exit code for this operation. `0` means success.
    """

    # Open the file as a tarfile
    try:
        with gzip.open(source, "rb") as hs:
            with open(target, "wb") as ht:
                # Get the length
                hs.seek(0, 2)
                total = hs.tell()
                hs.seek(0, 0)

                # Copy chunk-by-chunk
                done  = 0
                speed = 0
                print(f"Extracting '{source}' {bytes_to_string(done)}/{bytes_to_string(total)}... (0B/s)", end='\r')
                last_update = time.time()

                chunk = hs.read(65535)
                while len(chunk) > 0:
                    # Write the chunk
                    elapsed = time.time() - last_update
                    if elapsed >= 0.25:
                        print(elapsed)
                        print(1.0 / elapsed)
                        print((1.0 / elapsed) * speed)
                        print(f"Extracting '{source}' {bytes_to_string(done)}/{bytes_to_string(total)}... ({bytes_to_string((1.0 / elapsed) * speed)}/s)", end='\r')
                        last_update = time.time()
                        speed = 0
                    ht.write(chunk)
                    done  += len(chunk)
                    speed += len(chunk)

                    # Get the next chunk
                    chunk = hs.read(65535)
                print(f"Extracting '{source}' {bytes_to_string(done)}/{bytes_to_string(total)}... ({bytes_to_string((1.0 / (time.time() - last_update)) * speed)}/s)")

    except gzip.BadGzipFile as e:
        print(f"Failed to extract archive '{source}': {e}")
        return e.errno
    except IOError as e:
        print(f"Failed to extract archive '{source}': {e}")
        return e.errno

    # Done
    return 0

def remove_dir(target: str):
    """
        Removes the given directory and all its contents.

        # Arguments
        - `target`: The path to the directory to remove
    """

    try:
        for file in os.listdir(target):
            path = os.path.join(target, file)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                remove_dir(path)
        os.rmdir(target)
    except Exception as e:
        print(f"Failed to delete directory '{target}': {e}", file=sys.stderr)





##### ENTRYPOINT #####
def main() -> int:
    """
        Entrypoint to the script.

        # Returns
        The return code that the script should return.
    """

    # Create a temporary directory
    tmp_dir = f"/tmp/mnist_{''.join([random.choice(string.ascii_letters) for _ in range(8)])}"
    try:
        os.mkdir(tmp_dir)
    except IOError as e:
        print(f"Failed to create temporary directory '{tmp_dir}': {e}", file=sys.stderr)
        return e.errno

    # Attempt to download MNIST to there
    train_images = os.path.join(tmp_dir, "train-images-idx3-ubyte.gz")
    train_labels = os.path.join(tmp_dir, "train-labels-idx1-ubyte.gz")
    test_images  = os.path.join(tmp_dir, "t10k-images-idx3-ubyte.gz")
    test_labels  = os.path.join(tmp_dir, "t10k-labels-idx1-ubyte.gz")
    for (url, path) in [ ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", train_images), ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", train_labels), ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", test_images), ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", test_labels) ]:
        res = download(url, path)
        if res != 0:
            remove_dir(tmp_dir)
            return res

    # Extract the archives
    for (source, target) in [ (train_images, "./train-images.idx3-ubyte"), (train_labels, "./train-labels.idx1-ubyte"), (test_images, "./test-images.idx3-ubyte"), (test_labels, "./test-labels.idx1-ubyte") ]:
        res = unarchive(source, target)
        if res != 0:
            remove_dir(tmp_dir)
            return res

    # Don't forget the delete the directory
    remove_dir(tmp_dir)
    return 0



# Actual entrypoint
if __name__ == "__main__":
    # Define the arguments
    parser = argparse.ArgumentParser()

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    exit(main())

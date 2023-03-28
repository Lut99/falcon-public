#!/usr/bin/env python3
# SECRET SHARE.py
#   by Lut99
#
# Created:
#   27 Mar 2023, 18:25:00
# Last edited:
#   27 Mar 2023, 18:45:18
# Auto updated?
#   Yes
#
# Description:
#   A python script that secret-shares a list of numerical values into the
#   given number of parties.
#

import argparse
import random
import sys


##### ENTRYPOINT #####
def main(n_parties: int, files: list[str], file_type: str) -> int:
    # Iterateo throug hthe files
    for file in files:
        print(f"Secret-sharing '{file}' over {n_parties} parties...")

        # Open the output files
        out = [ open(file + "_" + (chr(ord('A') + i) if i < 26 else str(i)), "w") for i in range(n_parties) ]

        # Open-up the file
        try:
            with open(file, "r") as h:
                # Read a buffer for as long as we can
                buffer = ""
                run = True
                first = True
                while run:
                    chunk = h.read(4096)
                    if not chunk: break

                    # Iterate through the chunk to find numbers
                    for c in chunk:
                        if ord(c) >= ord('0') and ord(c) <= ord('9'):
                            buffer += c
                        elif c == ' ':
                            # Number boundary; we now have a number in the buffer

                            # Switch on what we expect
                            shares = []
                            if file_type == "uint8":
                                x = int(buffer)

                                # Find enough shares
                                for _ in range(n_parties):
                                    shares.append(random.randint(0, x))
                                    x -= shares[-1]
                            else:
                                print(f"ERROR: Unknown file type '{file_type}'", file=sys.stderr)
                                return 1

                            # Write the shares to the output files
                            for i in range(n_parties):
                                if first: first = False
                                else: out[i].write(' ')
                                out[i].write(str(shares[i]))

                            # Reset the buffer and continue
                            buffer = ""

                        else:
                            print(f"WARNING: File '{file}' contains unexpected character '{c}' (skipping file)", file=sys.stderr)
                            run = False
                            break

        except IOError as e:
            print(f"ERROR: Failed to open '{file}': {e} (skipping file)", file=sys.stderr)

        # Close the output files
        for o in out: o.close()

    # Do some flavour if we did nothing
    if len(files) == 0:
        print("No files given; nothing to do.")
    else:
        print("Done.")

    # Done!
    return 0



# Actual entrypoint
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("N_PARTIES", type=int, help="The number of parties to secret-share for.")
    parser.add_argument("FILES", nargs="*", help="The list of files to secret-share for.")
    parser.add_argument("-t", "--type", default="uint8", choices=["uint8"], help="The possible type of file that we expect in and expect out.")

    # Parse 'em
    args = parser.parse_args()

    # Call main
    exit(main(args.N_PARTIES, args.FILES, args.type))

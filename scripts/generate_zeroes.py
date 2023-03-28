#!/usr/bin/env python3
# GENERATE ZEROES.py
#   by Lut99
#
# Created:
#   29 Mar 2023, 00:06:18
# Last edited:
#   29 Mar 2023, 00:16:59
# Auto updated?
#   Yes
#
# Description:
#   Script that generates as much zeroes in a file, in another file.
#

import argparse
import sys


##### ENTYRPOINT #####
def main(input: str, output: str) -> int:
    # Open the input file
    zeroes = 0
    print("Writing zeroes...")
    try:
        with open(input, "r") as h_in:
            # Open the output file
            try:
                with open(output, "w") as h_out:
                    # Start iterating over the input...
                    first = True
                    buffer = ""
                    stop = False
                    while not stop:
                        # Read the next chunk from the file
                        chunk = h_in.read(65536)
                        if not chunk: break

                        # Parse the numbers in it
                        for c in chunk:
                            if c == ' ' or c == '\t' or c == '\r' or c == '\n':
                                # Skip empty sets (we are kind like that)
                                if len(buffer) > 0:
                                    # Parse the buffer as a float, altough we are kind and ignore if empty (i.e., two consecutive spaces)
                                    try:
                                        _ = float(buffer)
                                    except ValueError as e:
                                        print(f"Encountered illegal number '{buffer}': {e}", file=sys.stderr)
                                        return 1

                                    # Now write a zero for that
                                    if first: first = False
                                    else: h_out.write(" ")
                                    h_out.write("0")
                                    zeroes += 1

                                # Don't forget to reset the buffer
                                buffer = ""

                            else:
                                # Otherwise, keep appending to the buffer
                                buffer += c

            # Handle output file errors
            except IOError as e:
                print(f"Failed to open output file {output}: {e}", file=sys.stderr)
                return e.errno
    # Handle input file errors
    except IOError as e:
        print(f"Failed to open input file {input}: {e}", file=sys.stderr)
        return e.errno

    # Return the main as a success :)
    print(f"Done (written {zeroes} zeroes).")
    return 0



if __name__ == "__main__":
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help="The file to read the number of zeroes from (where we do one zero for every number, separated by spaces, we find)")
    parser.add_argument("OUTPUT", help="The file to write the new zeroes to.")

    # Parse 'em
    args = parser.parse_args()

    # Call main
    exit(main(args.INPUT, args.OUTPUT))

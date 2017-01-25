import os
import sys

BLOCK_STORAGE_DIR = sys.argv[1]

files = os.listdir(BLOCK_STORAGE_DIR)
for f in files:
    size = os.path.getsize(f)
    if size == 0:
        print("Deleting file {}".format(f))
        os.remove(f)

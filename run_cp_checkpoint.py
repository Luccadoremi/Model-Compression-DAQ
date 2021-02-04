import sys
import os
import argparse
import moxing as mox
import logging
import re
import fnmatch
import subprocess
import time

logging.basicConfig(level=logging.INFO)
# copy program to mox
# mox.file.copy_parallel("s3://mt-codes/marian", "/cache/marian")
# os.system("chmod +x /cache/marian/marian*")

os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"
mox.file.make_dirs("/cache")

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)


parser = argparse.ArgumentParser()
parser.add_argument("--train_url", type=str, default="")
args, _ = parser.parse_known_args()


# copy back to s3
model_dir = args.train_url
logging.info("copying output back...")
if not mox.file.exists(model_dir):
#     mox.file.remove(args.model_dir, recursive=True)
    mox.file.make_dirs(model_dir)

for i in range(360):
    time.sleep( 7200 )
    print('copy back checkpoint:',i)
    mox.file.copy_parallel(
        os.path.join(LOCAL_DIR, "model_dir"), 
        model_dir)

logging.info("end")
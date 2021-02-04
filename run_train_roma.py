import sys
import os
import argparse
import moxing as mox
import logging
import re
import fnmatch
import subprocess

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
parser.add_argument("--script", type=str, default="train-roma.sh")
parser.add_argument("--code_dir", type=str, default="s3://aarc-mt/code/fairseq_master")
parser.add_argument("--data_url", type=str, default="")
parser.add_argument("--file_pattern", type=str, default=None)
parser.add_argument("--train_url", type=str, default="")
parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument("--train_pref", type=str, default="")
parser.add_argument("--val_pref", type=str, default="")
parser.add_argument("--test_pref", type=str, default="")
parser.add_argument("--arch", type=str, default="our")
parser.add_argument("--config", type=str, default="")
parser.add_argument("--sub_config", type=str, default="")
parser.add_argument("--update_freq", type=str, default="1")

args, _ = parser.parse_known_args()

# copy to /cache
logging.info("copying " + args.code_dir)
mox.file.copy_parallel(
    args.code_dir, os.path.join(LOCAL_DIR,"code_dir"))

    
if not mox.file.exists(args.train_url):
#     mox.file.remove(args.model_dir, recursive=True)
    mox.file.make_dirs(args.train_url)
logging.info("copying " + args.train_url)
mox.file.copy_parallel(
    args.train_url, 
    os.path.join(LOCAL_DIR,"model_dir"))   

             
logging.info("copying data...")
if args.file_pattern:
    file_list=[]
    patterns = args.file_pattern.split(",")
    for fname in mox.file.list_directory(args.data_url, recursive=True):
        for pattern in patterns:
            if fnmatch.fnmatch(fname, pattern):
                file_list.append(fname)
                break
    logging.info(file_list)
    mox.file.copy_parallel(
        args.data_url, 
        os.path.join(LOCAL_DIR, "data_dir"),
        file_list=file_list)
else:
    logging.info(mox.file.list_directory(args.data_url, recursive=True))
    mox.file.copy_parallel(
        args.data_url, 
        os.path.join(LOCAL_DIR, "data_dir"))
             
# root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.join(LOCAL_DIR, "code_dir")+"/scripts"
script = os.path.join(root_dir, args.script)

logging.info("excuting ...")
cmd =["bash", script, args.train_url, args.arch, args.config, args.sub_config, args.gpu, args.update_freq]
process = subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout)
process.wait()

# copy back to s3
model_dir = args.train_url
logging.info("copying output back...")
if not mox.file.exists(model_dir):
#     mox.file.remove(args.model_dir, recursive=True)
    mox.file.make_dirs(model_dir)

mox.file.copy_parallel(
    os.path.join(LOCAL_DIR, "model_dir"), 
    model_dir)

logging.info("end")
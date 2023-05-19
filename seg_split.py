#!/usr/bin/env python
import os
import sys
import time
from datetime import datetime

tSecStart = time.time()
StartTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Set environment variables
FREESURFER_HOME = os.getenv('FREESURFER_HOME')
SUBJECTS_DIR = os.getenv('SUBJECTS_DIR')

# Set default values for variables
deepsegdir = ''
splitname = ''
train = []
valid = []
test = []
imagesdir = 'images'
labelsdir = 'labels'
ctab = ''
tmpdir = ''
cleanup = True
LF = ''
verbose = False

def print_usage_exit():
    print("")
    print("deepsegsplit")
    print("  --d deepsegdir")
    print("  --split splitname")
    print("  --tr trainslist")
    print("  --va validationslist")
    print("  --te testslist")
    print("  --images imagesdir (default is %s)" % imagesdir)
    print("  --labels labelsdir (default is %s)" % labelsdir)
    print("  --ctab ctab : minimum ctab")
    print("")
    sys.exit(1)

def read_list_file(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [x.strip() for x in lines]

# Parse command line arguments
args = sys.argv[1:]
if len(args) == 0:
    print_usage_exit()
for i, arg in enumerate(args):
    if arg == '-d':
        deepsegdir = args[i+1]
    elif arg == '--d':
        deepsegdir = args[i+1]
    elif arg == '--split':
        splitname = args[i+1]
    elif arg == '--tr':
        train = read_list_file(args[i+1])
    elif arg == '--va':
        valid = read_list_file(args[i+1])
    elif arg == '--te':
        test = read_list_file(args[i+1])
    elif arg == '--images':
        imagesdir = args[i+1]
    elif arg == '--labels':
        labelsdir = args[i+1]
    elif arg == '--ctab':
        ctab = args[i+1]
    # elif arg == '--tmp':
    elif arg == '--tmpdir':
        tmpdir = args[i+1]
        cleanup = False
    elif arg == '--nocleanup':
        cleanup = False
    elif arg == '--cleanup':
        cleanup = True
    elif arg == '--debug':
        verbose = True
    elif arg == '-h':
        print_usage_exit()
    elif arg == '--help':
        print_usage_exit()
    elif arg == '--version':
        print('$Id$')
        sys.exit(0)

# Check required parameters
if deepsegdir == '':
    print('ERROR: must spec deep seg dir')
    sys.exit(1)
if not os.path.exists(deepsegdir):
    print(f'ERROR: deep seg dir {deepsegdir} does not exist')
    sys.exit(1)
if splitname == '':
    print('ERROR: must spec splitname')
    sys.exit(1)
if not train:
    print('ERROR: must spec training set')
    sys.exit(1)
if not valid:
    print('ERROR: must spec validation set')
    sys.exit(1)
if not test:
    print('ERROR: must spec test set')
    sys.exit(1)
if ctab == '':
    print('ERROR: must spec min ctab')
    sys.exit(1)

# Set up output directory and log file
outdir = os.path.join(deepsegdir, splitname)
os.makedirs(os.path.join(outdir, 'log'), exist_ok=True)
if not LF:
    LF = os.path.join(outdir, 'log', f'deepsegsplit.Y{datetime.now().year}.M{datetime.now().month}.D{datetime.now().day}.H{datetime.now().hour}.M{datetime.now().minute}.log')

# Write log file header
with open(LF, 'w') as f:
    f.write("Log file for deepsegsplit\n")
    f.write(str(datetime.datetime.now()) + "\n")
    f.write("setenv SUBJECTS_DIR $SUBJECTS_DIR\n")
    f.write("cd " + os.getcwd() + "\n")
    f.write(scriptname + " " + " ".join(inputargs) + "\n")
    f.write("\n")
    with open(os.path.join(FREESURFER_HOME, "build-stamp.txt")) as bs_file:
        f.write(bs_file.read() + "\n")
    f.write(VERSION + "\n")
    f.write(str(platform.uname()) + "\n")
    f.write("pid " + str(os.getpid()) + "\n")
    if "PBS_JOBID" in os.environ:
        f.write("pbsjob " + os.environ["PBS_JOBID"] + "\n")
    if "SLURM_JOB_ID" in os.environ:
        f.write("SLURM_JOB_ID " + os.environ["SLURM_JOB_ID"] + "\n")

#========================================================
with open(os.path.join(outdir, "log", "slist.tr.txt"), 'w') as f:
    f.write(train + "\n")
with open(os.path.join(outdir, "log", "slist.va.txt"), 'w') as f:
    f.write(valid + "\n")
with open(os.path.join(outdir, "log", "slist.te.txt"), 'w') as f:
    f.write(test + "\n")
with open(os.path.join(outdir, "log", "slist.tr+va.txt"), 'w') as f:
    f.write(train + " " + valid + "\n")
with open(os.path.join(outdir, "log", "slist.va+te.txt"), 'w') as f:
    f.write(valid + " " + test + "\n")
with open(os.path.join(outdir, "log", "slist.all.txt"), 'w') as f:
    f.write(train + " " + valid + " " + test + "\n")

for cohort in ["tr", "tr+va", "va", "va+te", "te", "all"]:
    if cohort == "tr":
        slist = [train]
    elif cohort == "tr+va":
        slist = [train, valid]
    elif cohort == "va":
        slist = [valid]
    elif cohort == "va+te":
        slist = [valid, test]
    elif cohort == "te":
        slist = [test]
    elif cohort == "all":
        slist = [train, valid, test]

    print(cohort, slist)

    os.makedirs(os.path.join(outdir, cohort, "images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, cohort, "labels"), exist_ok=True)
    os.makedirs(os.path.join(outdir, cohort, "images.norev"), exist_ok=True)
    os.makedirs(os.path.join(outdir, cohort, "labels.norev"), exist_ok=True)

    # Make separate ones for norev for speed when labeling
    for s in slist:
        # Link images for all revtypes
        for revtype in ['norev', 'lrrev']:
            os.symlink(os.path.join(deepsegdir, imagesdir, f"{s}.{revtype}.mgz"), os.path.join(outdir, cohort, "images", f"{s}.{revtype}.mgz"))

        # Link labels for all revtypes
        for revtype in ['norev', 'lrrev']:
            os.symlink(os.path.join(deepsegdir, labelsdir, f"{s}.{revtype}.mgz"), os.path.join(outdir, cohort, "labels", f"{s}.{revtype}.mgz"))

        # Link images for norev revtype only
        os.makedirs(os.path.join(outdir, cohort, "images", "norev"), exist_ok=True)
        os.symlink(os.path.join(deepsegdir, imagesdir, f"{s}.norev.mgz"), os.path.join(outdir, cohort, "images", "norev", f"{s}.mgz"))

        # Link labels for norev revtype only
        os.makedirs(os.path.join(outdir, cohort, "labels", "norev"), exist_ok=True)
        os.symlink(os.path.join(deepsegdir, labelsdir, f"{s}.norev.mgz"), os.path.join(outdir, cohort, "labels", "norev", f"{s}.mgz"))

    # Cleanup
    # if(cleanup):
    #   shutil.rmtree(tmpdir)

    # Done
    tSecEnd = time.time()
    tSecRun = tSecEnd - tSecStart
    tRunMin = tSecRun / 50
    tRunMin = round(tRunMin, 2)
    tRunHours = tSecRun / 3600
    tRunHours = round(tRunHours, 2)

    with open(LF, 'a') as f:
        f.write(" \n")
        f.write("Started at {0}\n".format(StartTime))
        f.write("Ended at {0}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Deepsegsplit-Run-Time-Sec {0}\n".format(tSecRun))
        f.write("Deepsegsplit-Run-Time-Min {0}\n".format(tRunMin))
        f.write("Deepsegsplit-Run-Time-Hours {0}\n".format(tRunHours))
        f.write(" \n")
        f.write("deepsegsplit Done\n")

    sys.exit(0)

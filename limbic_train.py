#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import subprocess
import time
import textwrap

VERSION = '$Id$'
scriptname = os.path.basename(sys.argv[0])
os.environ['SUBJECTS_DIR'] = '/autofs/cluster/fsm/users/greve/subjects'
os.environ['FS_SBATCH_ACCOUNT'] = 'lcnrtx'
os.environ['FS_SBATCH_ACCOUNT'] = 'fsm'
part = 'rtx6000,rtx8000'

trainpy = '/homes/4/greve/l/sp1/hthsuseg-billot/scripts/training.py'
submit = False
jobname = None

imagedir = None
modeldir = None
labeldir = None
labellist = None
DoAll = 0
DoConf = 0
DoTrain = 0
DoTest = 0
DoValid = 0
cudavisibledevice = None
wl2_epochs = 5
steps_per_epoch = 1000
rlflip = 0
n_neutral_labels = 2
AugAug = 0
InitModel = None
Resume = 0
log = None
epoch = 0

convsize = 3
nlevels = 3
featmult = 2
convperlevel = 2
actfunc = 'elu'
lrate = 1e-4
lratedecay = 1e-6
diceepochs = 200
unetshape = 160
unetfeatures = 24
batchsize = 1
noisestd = 0
os.environ['DEEP_LIMBIC_SEED'] = '123'

tmpdir = None
cleanup = 1
LF = None

inputargs = sys.argv[1:]
PrintHelp = False
n = sum(arg == '-help' for arg in inputargs)
if n != 0:
    PrintHelp = True

n = sum(arg == '-version' for arg in inputargs)
if n != 0:
    print(VERSION)
    sys.exit(0)

# TODO: Add the command line arguments parsing logic here

# TODO: Add the command line arguments checking logic here

if submit:
    cmd = ['deeplimbictrain'] + inputargs + ['--no-submit']
    sbatch = ['sbatch', '--partition={}'.format(part), '--nodes=1', '--account=fhs', '--gpus=1', '--ntasks-per-node=1',
              '--cpus-per-task=3']
    sbatch += ['--time=3-0']
    model = os.path.basename(modeldir)
    os.makedirs('/space/sulc/1/users/greve/sbatchlog', exist_ok=True)
    if not jobname:
        jobname = model
    batchlogdir = os.path.join(os.environ['SBD'], 'log', os.uname().nodename)
    os.makedirs(batchlogdir, exist_ok=True)
    if not log:
        log = os.path.join(batchlogdir, jobname)
    os.remove(log)
    cmd = sbatch + ['--job-name={}'.format(jobname), '--output={}'.format(log)] + cmd
    print(cmd)
    subprocess.run(cmd)
    sys.exit(0)

StartTime = time.strftime("%c")
tSecStart = time.time()
year = time.strftime("%Y")
month = time.strftime("%m")
day = time.strftime("%d")
hour = time.strftime("%H")
minute = time.strftime("%M")

os.makedirs(os.path.join(modeldir, 'log', 'log'), exist_ok=True)
os.chdir(modeldir)

if not tmpdir:
    if os.path.exists('/scratch'):
        tmpdir = f'/scratch/tmpdir.deeplimbictrain.{os.getpid()}'
    else:
        tmpdir = f'{modeldir}/tmpdir.deeplimbictrain.{os.getpid()}'

if not LF:
    LF = f'{modeldir}/log/log/deeplimbictrain.Y{year}.M{month}.D{day}.H{hour}.M{minute}.log'
if LF != '/dev/null':
    os.remove(LF)
    
with open(LF, 'a') as log_file:
    log_file.write("Log file for deeplimbictrain\n")
    log_file.write(time.strftime("%c") + "\n\n")
    log_file.write(f"setenv SUBJECTS_DIR {os.environ['SUBJECTS_DIR']}\n")
    log_file.write(f"cd {os.getcwd()}\n")
    log_file.write(" ".join([sys.argv[0]] + inputargs) + "\n\n")
    log_file.write("TODO: Read and print FREESURFER_HOME/build-stamp.txt contents\n")
    log_file.write(VERSION + "\n")
    log_file.write(" ".join(os.uname()) + "\n")
    log_file.write(f"pid {os.getpid()}\n")
    
    if 'PBS_JOBID' in os.environ:
        log_file.write(f"pbsjob {os.environ['PBS_JOBID']}\n")
    
    if 'SLURM_JOB_ID' in os.environ:
        log_file.write(f"SLURM_JOB_ID {os.environ['SLURM_JOB_ID']}\n")
        log_file.write(" ".join(os.uname()) + "\n")
        log_file.write("TODO: Read and print environment variables containing CUDA and GPU\n")
        log_file.write("TODO: Run nvidia-smi command and capture output\n")

### Part 2
cfgfile = os.path.join(modeldir, 'log', 'train.config.dat')

with open(cfgfile, 'w') as config_file:
    config_file.write(f"convsize {convsize}\n")
    config_file.write(f"nlevels {nlevels}\n")
    config_file.write(f"featmult {featmult}\n")
    config_file.write(f"convperlevel {convperlevel}\n")
    config_file.write(f"actfunc {actfunc}\n")
    config_file.write(f"lrate {lrate}\n")
    config_file.write(f"lratedecay {lratedecay}\n")
    config_file.write(f"diceepochs {diceepochs}\n")
    config_file.write(f"wl2epochs {wl2_epochs}\n")
    config_file.write(f"steps_per_epoch {steps_per_epoch}\n")
    config_file.write(f"rlflip {rlflip}\n")
    config_file.write(f"n_neutral_labels {n_neutral_labels}\n")
    config_file.write(f"unetshape {unetshape}\n")
    config_file.write(f"unetfeatures {unetfeatures}\n")
    config_file.write(f"batchsize {batchsize}\n")
    config_file.write(f"noisestd {noisestd}\n")
    config_file.write(f"DEEP_LIMBIC_SEED {os.environ['DEEP_LIMBIC_SEED']}\n")
    config_file.write(f"AugAug {AugAug}\n")
    if not Resume:
        config_file.write(f"InitModel {InitModel}\n")

trainpy_cmd = [
    trainpy, imagedir, labeldir, modeldir,
    "--save_label_list", os.path.join(modeldir, "labels.npy"),
    "--conv_size", str(convsize), "--n_levels", str(nlevels),
    "--feat_mult", str(featmult), "--conv_per_level", str(convperlevel),
    "--activation", actfunc, "--lr", str(lrate), "--lr_decay", str(lratedecay),
    "--wl2_epochs", str(wl2_epochs), "--dice_epochs", str(diceepochs),
    "--steps_per_epoch", str(steps_per_epoch), "--unet_features", str(unetfeatures),
    "--batchsize", str(batchsize), "--noise_std", str(noisestd)
]

if rlflip:
    trainpy_cmd.extend(["--flip_rl_only", "--n_neutral_labels", str(n_neutral_labels)])
else:
    trainpy_cmd.append("--no_flip")

if labellist:
    trainpy_cmd.extend(["--label_list", labellist])

if unetshape > 0:
    trainpy_cmd.extend(["--unet_shape", str(unetshape)])

if AugAug != 0:
    trainpy_cmd.extend([
        "--scaling", "0.2", "--shearin", ".024",
        "--nonlin_std", "4.5", "--nonlin_shape_factor", ".08"
    ])

if InitModel != 'None':
    trainpy_cmd.extend(["--wl2_epochs", "0", "--load_model_file", InitModel])

if Resume:
    trainpy_cmd.extend(["--initial_epoch_dice", str(epoch)])

with open(LF, 'a') as log_file:
    log_file.write("\n")
    log_file.write(" ".join(trainpy_cmd) + "\n")
    log_file.write("\n")

# TODO: Replace "fs_time python" with appropriate Python command
subprocess.run(["fs_time", "python"] + trainpy_cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

# Cleanup
# if cleanup: shutil.rmtree(tmpdir)

# Done
tSecEnd = int(time.time())
tSecRun = tSecEnd - tSecStart
tRunMin = tSecRun / 60
tRunHours = tSecRun / 3600

with open(LF, 'a') as log_file:
    log_file.write("\n")
    log_file.write(f"Started at {StartTime}\n")
    log_file.write(f"Ended   at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Deeplimbictrain-Run-Time-Sec {tSecRun}\n")
    log_file.write(f"Deeplimbictrain-Run-Time-Min {tRunMin:.2f}\n")
    log_file.write(f"Deeplimbictrain-Run-Time-Hours {tRunHours:.2f}\n")
    log_file.write("\n")
    log_file.write("deeplimbictrain Done\n")

### Part 3
def parse_args(args):
    parser = argparse.ArgumentParser(description="Deeplimbic Train")

    parser.add_argument("--m", required=True, help="Model directory")
    parser.add_argument("--i", required=True, help="Image directory")
    parser.add_argument("--l", required=True, help="Label directory")
    parser.add_argument("--partition", type=int, help="Partition")
    parser.add_argument("--conv_size", type=int, help="Convolution size")
    parser.add_argument("--n_levels", type=int, help="Number of levels")
    parser.add_argument("--feat_mult", type=float, help="Feature multiplier")
    parser.add_argument("--conv_per_level", type=int, help="Convolution per level")
    parser.add_argument("--activation", help="Activation function")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, help="Learning rate decay")
    parser.add_argument("--dice_epochs", type=int, help="Dice epochs")
    parser.add_argument("--wl2_epochs", type=int, help="Weighted L2 epochs")
    parser.add_argument("--steps_per_epoch", type=int, help="Steps per epoch")
    parser.add_argument("--rlflip", action='store_true', help="Right-left flip")
    parser.add_argument("--augaug", action='store_true', help="Augmentation")
    parser.add_argument("--batchsize", type=int, help="Batch size")
    parser.add_argument("--noisestd", type=float, help="Noise standard deviation")
    parser.add_argument("--unet_shape", type=int, help="UNET shape")
    parser.add_argument("--unet_features", type=int, help="UNET features")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--c", "--cuda", type=int, help="CUDA visible device")
    parser.add_argument("--cpu", action='store_true', help="Use CPU instead of GPU")
    parser.add_argument("--jobname", help="Job name")
    parser.add_argument("--submit", action='store_true', help="Submit")
    parser.add_argument("--no-submit", action='store_true', help="No submit")
    parser.add_argument("--resume", action='store_true', help="Resume")
    parser.add_argument("--lf", help="Log file")
    parser.add_argument("--log", help="Log")
    parser.add_argument("--init-model", help="Initial model")
    parser.add_argument("--nolog", "--no-log", action='store_true', help="No log")
    parser.add_argument("--tmp", "--tmpdir", help="Temporary directory")
    parser.add_argument("--nocleanup", action='store_true', help="No cleanup")
    parser.add_argument("--cleanup", action='store_true', help="Cleanup")
    parser.add_argument("--debug", action='store_true', help="Debug")

    parsed_args = parser.parse_args(args)
    return parsed_args

### Part 4
def check_params(args):
    if args.m is None:
        print("ERROR: must specify modeldir")
        sys.exit(1)
    if args.i is None:
        print("ERROR: must specify imagedir")
        sys.exit(1)
    if args.l is None:
        print("ERROR: must specify labeldir")
        sys.exit(1)

    if args.rlflip:
        labellist = os.path.join(args.labeldir, 'labels.lrsorted.npy')

    if args.labellist is not None:
        if not os.path.exists(labellist):
            print(f"ERROR: cannot find {labellist}")
            sys.exit(1)

    if args.resume:
        cfg = os.path.join(args.modeldir, 'log', 'train.config.dat')
        if not os.path.exists(cfg):
            print(f"ERROR: resume: cannot find {cfg}")
            sys.exit(1)

        # Read parameters from the config file
        with open(cfg, 'r') as f:
            config_lines = f.readlines()

        config_dict = {}
        for line in config_lines:
            key, value = line.strip().split()
            config_dict[key] = value

        # Update args with values from the config file
        for key, value in config_dict.items():
            setattr(args, key, value)

        # Find the most recent model file for resuming training
        epoch = 300
        while True:
            epochstr = f"{epoch:03d}"
            init_model = os.path.join(args.modeldir, f"dice_{epochstr}.h5")
            if os.path.exists(init_model):
                break
            if epoch == 0:
                init_model = os.path.join(args.modeldir, "wl2_005.h5")
                if os.path.exists(init_model):
                    break
                print("ERROR: resume: cannot find model")
                sys.exit(1)
            epoch -= 1

        if epoch == args.dice_epochs:
            print(f"Model has already been trained to {args.dice_epochs}")
            sys.exit(1)

        print(f"Resume: init model is {init_model}, epoch {epoch}")
        args.epoch = epoch + 1

    return args

### Part 5
def usage_exit(print_help=False):
    message = textwrap.dedent(f"""
    deeplimbictrain
     --i imagedir
     --l labeldir
     --m modeldir
     --c cudavisibledevice
     --conv_size convsize
     --conv_per_level convperlevel
     --n_levels  nlevels
     --feat_mult featmult
     --lr learningrate
     --lr_decay learningratedecay
     --unet_shape shape
     --batchsize batchsize
     --augaug : enhanced augmentation
     --noisestd nstd
     --init-model model.h5
     --submit : submit to sbatch
     --resume : given the model dir, resume where it left off
     --jobname jobname : submit to sbatch with jobname
     --partition partition
    """)

    print(message)

    if not print_help:
        sys.exit(1)

    print("BEGINHELP")
    print("\ncrop003")
    print("73 x 51 x 70 LIA")
    print("Image shape  (73, 70, 51, 15)")
    print("Croppping    [73, 70, 51]")
    print("             [72, 64, 48] (72, 64, 48, 15)")

    print("\ncrop006")
    print("Image shape  (79, 76, 57, 15)")
    print("Croppping  [79, 76, 57]")
    print("           [72, 72, 56] (72, 72, 56, 15)")
    print("Output shape  [72, 72, 56]")

    print("\ntensorboard --logdir=model.train.default.a/logs")
    sys.exit(1)




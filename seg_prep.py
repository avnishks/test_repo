import argparse
import os
import sys


def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--o', dest='outdir', type=str, required=True, help='The output directory.')
    parser.add_argument('--i', dest='inputvol', type=str, required=True, help='The input volume file.')
    parser.add_argument('--seg', dest='manseg', type=str, required=True, help='The manual segmentation file.')
    parser.add_argument('--ctab', dest='ctab', type=str, required=True, help='The color table file.')
    parser.add_argument('--s', dest='subjects', type=str, nargs='+', help='The list of subjects.')
    parser.add_argument('--f', dest='subject_files', type=str, help='The file containing the list of subjects.')
    parser.add_argument('--left', dest='leftseglist', type=str, nargs='+', help='The list of left hemisphere segmentation labels.')
    parser.add_argument('--right', dest='rightseglist', type=str, nargs='+', help='The list of right hemisphere segmentation labels.')
    parser.add_argument('--hipamyg', dest='hipamyg', action='store_true', help='Use the hippocampus and amygdala segmentation labels.')
    parser.add_argument('--other', dest='otherseglist', type=str, nargs='+', help='The list of other segmentation labels.')
    parser.add_argument('--labeldir', dest='labeldir', type=str, default='mri', help='The directory containing the segmentation labels.')
    parser.add_argument('--sd', dest='subjects_dir', type=str, help='The directory containing the subjects.')
    parser.add_argument('--allow-skip', dest='allow_skip', action='store_true', help='Allow skipping of subjects that do not exist.')
    parser.add_argument('--no-rescale', dest='rescale', action='store_false', help='Do not rescale the output images.')
    parser.add_argument('--log', dest='logfile', type=str, help='The log file.')
    parser.add_argument('--nolog', dest='logfile', action='store_false', help='Do not write a log file.')
    parser.add_argument('--tmp', dest='tmpdir', type=str, help='The temporary directory.')
    parser.add_argument('--nocleanup', dest='cleanup', action='store_false', help='Do not clean up the temporary directory.')
    args = parser.parse_args()

    # Check the command line arguments.
    if not os.path.isdir(args.outdir):
        print('Error: The output directory does not exist.')
        sys.exit(1)
    if not os.path.isfile(args.inputvol):
        print('Error: The input volume file does not exist.')
        sys.exit(1)
    if not os.path.isfile(args.seg):
        print('Error: The manual segmentation file does not exist.')
        sys.exit(1)
    if not os.path.isfile(args.ctab):
        print('Error: The color table file does not exist.')
        sys.exit(1)
    if args.subjects and not args.subject_files:
        print('Error: The list of subjects must be specified with either the --subjects or --subject_files argument.')
        sys.exit(1)
    # if args.subjects and not all(os.path.isdir(os.path.join(args.subjects_dir, subject)) for subject in args.subjects):
    #     print('Error: One or more subjects do not exist.')
#######
    if args.subjects and not all(os.path.isdir(os.path.join(args.subjects_dir, subject)) for subject in args.subjects):
        print('Error: One or more subjects do not exist.')
        sys.exit(1)

    # Create the output directory.
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Create the log file.
    if args.logfile:
        with open(args.logfile, 'w') as f:
            f.write('Log file for deepsegprep\n')
            f.write('Date: ' + str(datetime.now()) + '\n')
            f.write('Arguments: ' + str(args) + '\n')

    # Get the training and validation subjects.
    if args.subjects:
        train_subjects = args.subjects
    else:
        train_subjects = []
    if args.subject_files:
        with open(args.subject_files) as f:
            for line in f:
                subject = line.strip()
                if subject not in train_subjects:
                    train_subjects.append(subject)

    # Create the training and validation directories.
    train_dir = os.path.join(args.outdir, 'train')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    val_dir = os.path.join(args.outdir, 'val')
    if not os.path.isdir(val_dir):
        os.makedirs(val_dir)

    # Create the training and validation datasets.
    train_dataset = ImageDataset(train_dir, args.inputvol, args.seg, args.ctab, args.leftseglist, args.rightseglist, args.hipamyg, args.otherseglist, args.rescale, args.allow_skip)
    val_dataset = ImageDataset(val_dir, args.inputvol, args.seg, args.ctab, args.leftseglist, args.rightseglist, args.hipamyg, args.otherseglist, args.rescale, args.allow_skip)

    # Train the network.
    model = DeepSegNet(args.labeldir)
    model.train(train_dataset, val_dataset, args.epochs, args.batch_size, args.lr, args.momentum, args.decay, args.logfile)

    # Save the network.
    model.save(args.outdir)

    # Print the results.
    print('Training finished.')
    print('The network has been saved to {}.'.format(args.outdir))

if __name__ == '__main__':
    main()

import os
import random
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Split files from a folder into test, train, and validation sets.")
parser.add_argument("folder", type=str, help="Path to the folder containing the files")

# Parse the command-line arguments
args = parser.parse_args()

folder_path = args.folder  # Get the folder path from the command-line argument
test_percentage = 0.2  # Percentage of files for the test set
train_percentage = 0.6  # Percentage of files for the train set
valid_percentage = 0.2  # Percentage of files for the validation set

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Shuffle the file list randomly
random.shuffle(file_list)

# Calculate the number of files for each set
total_files = len(file_list)
test_size = int(total_files * test_percentage)
train_size = int(total_files * train_percentage)
valid_size = int(total_files * valid_percentage)

# Split the file list into test, train, and validation sets
test_set = [os.path.splitext(os.path.splitext(file_name)[0])[0] for file_name in file_list[:test_size]]
train_set = [os.path.splitext(os.path.splitext(file_name)[0])[0] for file_name in file_list[test_size:test_size + train_size]]
valid_set = [os.path.splitext(os.path.splitext(file_name)[0])[0] for file_name in file_list[test_size + train_size:test_size + train_size + valid_size]]

# Write the filenames to the respective text files
with open("test1.txt", "w") as test_file:
    test_file.write("\n".join(test_set))

with open("train1.txt", "w") as train_file:
    train_file.write("\n".join(train_set))

with open("valid1.txt", "w") as valid_file:
    valid_file.write("\n".join(valid_set))

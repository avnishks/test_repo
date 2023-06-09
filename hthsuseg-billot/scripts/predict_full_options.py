import site
import sys
site.addsitedir('/autofs/space/iddhi_001/users/greve/hthsuseg-billot');
print(sys.path)
import os

from argparse import ArgumentParser
from hypothalamus_seg.predict import predict

parser = ArgumentParser()

# positional arguments
parser.add_argument("path_images", type=str,
                    help="images to segment. Can be the path to a single image or to a folder.")
parser.add_argument("path_segmentations", type=str,
                    help="saving path. Can be a single image (if path_images is an image), or a folder.")
parser.add_argument("path_model", type=str, help="path to a h5 file containing the trained model.")
parser.add_argument("segmentation_labels", type=str,
                    help="path to a numpy array containing all the segmentation label values")

# preprocessing/postprocessing parameters
parser.add_argument("--padding", type=int, dest="padding", default=None,
                    help="margin to pad images with before prediction. Default is no padding.")
parser.add_argument("--cropping", type=int, dest="cropping", default=None,
                    help="crop images to this size before prediction. Default is no cropping.")
parser.add_argument("--resample", type=float, dest="resample", default=None,
                    help="resolution at which to resample the images before prediction. "
                         "Segmentations will then be resampled back to native resolution. Default is no resampling.")
parser.add_argument("--smoothing", type=float, dest="sigma_smoothing", default=0,
                    help="std dev of Gaussian kernel to smooth the predictions.")
parser.add_argument("--biggest_component", action='store_true', dest="keep_biggest_component",
                    help="whether to only keep the biggest component (linked voxels) of the segmentation")

# saving paths
parser.add_argument("--out_posteriors", type=str, default=None, dest="path_posteriors",
                    help="path to a folder where to save the output posteriors")
parser.add_argument("--out_volumes", type=str, default=None, dest="path_volumes",
                    help="path to a csv file where to save the volumes of all subunits for all patients.")
parser.add_argument("--include_background_volume", action='store_false', dest="skip_background_volume",
                    help="whether to include the volume of the background if out_volumes is provided")

# Architecture parameters (should be the same as training)
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3, help="size of convolution masks")
parser.add_argument("--n_levels", type=int, dest="n_levels", default=3, help="number of level")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2, help="convolution per level")
parser.add_argument("--unet_features", type=int, dest="unet_feat_count", default=24, help="features of first layer")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2, help="feature multiplication ratio")
#parser.add_argument("--no_batch_norm", action='store_true', dest="no_batch_norm", help="deactivate batch normalisation")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function")

# Evaluation parameters
parser.add_argument("--gt_folder", type=str, default=None, dest="gt_folder",
                    help="folder with ground truth segmentations for evaluation.")

args = parser.parse_args()
predict(**vars(args))

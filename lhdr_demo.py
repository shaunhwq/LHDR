import os
import time
from os import path
import argparse
import numpy as np
import torch
import cv2
from network import LiteHDRNet
from tqdm import tqdm


def pyr_crop(img: np.array, num_layers=3) -> np.array:
    """
    Prevents errors for pyramid style networks by center cropping such that dimensions are divisible by num_layers power of 2.

    :param img: input image to be cropped
    :param num_layers: Crop input image to be able to fit a network with pyramid with num_layers layers.
    :returns: cropped image.
    """
    h, w, _ = img.shape
    des_h, des_w = np.floor(np.array([h, w]) / pow(2, num_layers - 1)).astype(np.int32) * pow(2, num_layers - 1)
    w_start, h_start = (w - des_w) // 2, (h - des_h) // 2
    return img[h_start: h_start + des_h, w_start: w_start + des_w, ::]


def restore_pyr_crop(img: np.array, original_shape: tuple) -> np.array:
    """
    Zero pads an image that has been cropped to its original size

    :param img: Cropped image (using pyr_crop for example)
    :param original_shape: Original shape of the image before it was cropped in the format (h, w, c)
    :returns: Zero padded img with size original_shape
    """
    oh, ow, _ = original_shape
    ih, iw, _ = img.shape
    if oh == ih and ow == iw:
        return img

    # Zero pad and place cropped image in center
    restored = np.zeros(original_shape, dtype=np.float32)
    start_h, start_w = int((oh - ih) // 2), int((ow - iw) // 2)
    restored[start_h: start_h + ih, start_w: start_w + iw, ::] = img

    return restored


### System utilities ###
def process_path(directory, create=False):
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


def compose(transforms):
    """Composes list of transforms (each accept and return one item)"""
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), "list of functions expected"

    def composition(obj):
        """Composite function"""
        for transform in transforms:
            obj = transform(obj)
        return obj
    return composition


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


### Image utilities ###
def np2torch(img):
    img = img[:, :, [2, 1, 0]]
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
# def np2torch(np_img):
#     rgb = np_img[:, :, (2, 1, 0)]
#     return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2np(t_img):
    img_np = t_img.detach().numpy()
    return np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)).astype(np.float32)
# def torch2np(t_img):
#     return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


class Exposure(object):
    def __init__(self, stops, gamma):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img*(2**self.stops), 0, 1)**self.gamma


### Parameters ###
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg("--input_dir", help="Path to input dir", required=True, type=str)
arg("--output_dir", help="Path to output dir", required=True, type=str)
arg("--device", help="device", default="cuda:0", type=str)
arg('-out_format', choices=['hdr', 'exr', 'png'], default='hdr', help='Encapsulation of output HDR image.')
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

### Load network ###
net = LiteHDRNet(in_nc=3, out_nc=3, nf=32, act_type='leakyrelu')
net.load_state_dict(torch.load('params.pth', map_location=lambda s, l: s))
net.to(opt.device)
net.eval()

### Loading images ###
preprocess = compose([lambda x: x.astype('float32')])
image_paths = [os.path.join(opt.input_dir, img_path) for img_path in os.listdir(opt.input_dir) if img_path[0] != "."]


for ldr_file in tqdm(image_paths, total=len(image_paths), desc="Running LHDR..."):
    in_image = cv2.imread(ldr_file, cv2.IMREAD_UNCHANGED)
    loaded = pyr_crop(in_image.copy(), num_layers=3)
    start = time.time()
    ldr_input = preprocess(loaded) / 255.0

    # copy input numpy to [img, s_cond, c_cond] to suit the network model
    s_cond_prior = ldr_input.copy()
    s_cond_prior = np.clip((s_cond_prior - 0.9)/(1 - 0.9), 0, 1)  # now masked outside the network
    c_cond_prior = cv2.resize(ldr_input.copy(), (0, 0), fx=0.25, fy=0.25)

    ldr_input_t = np2torch(ldr_input).unsqueeze(dim=0)
    s_cond_prior_t = np2torch(s_cond_prior).unsqueeze(dim=0)
    c_cond_prior_t = np2torch(c_cond_prior).unsqueeze(dim=0)

    ldr_input_t = ldr_input_t.to(opt.device)
    s_cond_prior_t = s_cond_prior_t.to(opt.device)
    c_cond_prior_t = c_cond_prior_t.to(opt.device)

    x = (ldr_input_t, s_cond_prior_t, c_cond_prior_t)

    with torch.no_grad():
        prediction = net(x)

    prediction = prediction.detach()[0].float().cpu()
    prediction = torch2np(prediction)

    prediction = prediction / prediction.max()

    #prediction = prediction ** 2.24

    prediction = restore_pyr_crop(prediction, in_image.shape)

    new_name = os.path.splitext(os.path.basename(ldr_file))[0] + ".hdr"
    output_path = os.path.join(opt.output_dir, new_name)
    cv2.imwrite(output_path, prediction)

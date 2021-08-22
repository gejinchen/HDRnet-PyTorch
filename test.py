import numpy as np
import rawpy
import torch
from argparse import ArgumentParser
from models import HDRnetModel
from PIL import Image
from skimage import io, transform
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from utils import psnr, load_test_ckpt


def test(params, model):
    if params['cuda']:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if params['hdr']:
            low, full = load_img_hdr(params)
        else:
            low, full = load_img(params)

        low = low.to(device)
        full = full.to(device)
        # Normalize to [0, 1] on GPU
        if params['hdr']:
            low = torch.div(low, 65535.0)
            full = torch.div(full, 65535.0)
        else:
            low = torch.div(low, 255.0)
            full = torch.div(full, 255.0)

        output = model(low, full)

        output = torch.clamp(output, 0, 1)
        save_image(output, 'output.png')
        save_image(full, 'input.png')

def load_img(params):
    full = io.imread(params['test_path'])
    full = torch.from_numpy(full.transpose((2, 0, 1)))
    low = resize(full, (params['input_res'], params['input_res']), Image.NEAREST)
    low = low.unsqueeze(0)
    full = full.unsqueeze(0)
    return low, full

def load_img_hdr(params):
    full = rawpy.imread(params['test_path'])
    full = full.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    full = np.asarray(full, dtype=np.float32)
    full = torch.from_numpy(full.transpose((2, 0, 1)))
    low = resize(full, (params['input_res'], params['input_res']), Image.NEAREST)

    low = low.unsqueeze(0)
    full = full.unsqueeze(0)
    return low, full

def parse_args():
    parser = ArgumentParser(description='HDRnet testing')
    parser.add_argument('--test_path', type=str, required=True, help='Test image path')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse test parameters
    new_params = vars(parse_args())

    # Test model
    state_dict, params = load_test_ckpt(new_params['ckpt_path'])
    params.update(new_params)
    model = HDRnetModel(params)
    model.load_state_dict(state_dict)

    test(params, model)
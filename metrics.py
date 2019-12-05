
import numpy as np
import PIL.Image as Image
import skvideo.measure.niqe as niqe
from sewar.full_ref import psnr, ssim


def get_metrics(GT, reconstructed, mode=False):
    if mode:
        pass
    _psnr = psnr(np.array(GT), np.array(reconstructed))
    _ssim = ssim(np.array(GT), np.array(reconstructed))
    #_niqe = niqe(np.array(reconstructed))
    return _psnr, _ssim[0], 0
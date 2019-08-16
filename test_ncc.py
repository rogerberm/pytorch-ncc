"""
(comment here)

roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

import torch
from NCC import NCC
from matplotlib import pyplot as plt
import numpy as np

def test_ncc():
    lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1)
    patch_center = 275, 275
    y1, y2 = patch_center[0] - 25, patch_center[0] + 25
    x1, x2 = patch_center[1] - 25, patch_center[1] + 25
    lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
    ncc = NCC(lena_patch)
    ncc_response = ncc(lena_tensor[None, ...])
    assert ncc_response.max().isclose(torch.ones(1))
    assert np.unravel_index(ncc_response.argmax(), lena_tensor.shape) == (0, 275, 275)

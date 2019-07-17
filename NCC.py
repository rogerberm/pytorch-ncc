"""
Normalized Cross-Correlation for pattern matching.
Torch implementation

roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

import logging
import torch


ncc_logger = logging.getLogger(__name__)


class PatchStd(torch.nn.Module):
    """
    Computes standard deviations within a local window (i.e., convolution-style)

    kernel_size controls the size of the local neighborhood.
    """
    def __init__(self, kernel_size):
        super().__init__()

        dimensions = len(kernel_size)
        assert dimensions in (1, 2, 3), f"Invalid template dimensions {dimensions}. Only 1, 2, or 3 dimensions allowed."

        if any(side % 2 == 0 for side in kernel_size):
            ncc_logger.warning("Might break with even kernel sizes. Kernel size is %s.", kernel_size)

        padding = tuple(side//2 for side in kernel_size)

        # Setup convolution to compute local means.
        conv_module = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)[dimensions - 1]
        self._patch_mean = conv_module(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self._patch_mean.weight.data.fill_(1 / self._patch_mean.weight.numel())

    def forward(self, image):
        # Compute standard deviation in terms of means: std = E[X^2] - E[X]^2.
        result = self._patch_mean(image**2) - self._patch_mean(image)**2
        result = result.sqrt()

        return result


class NCC(torch.nn.Module):
    """
    Computes the Normalized Cross-Correlation of an image to a template.

    NCC(I, T) = (1 / sigma_I) * corr(I, T_tilde)
        where sigma_I contains the local standard deviations of the image,
        and T_tilde is the normalized template.
    """
    def __init__(self, template):
        super().__init__()

        self._std = PatchStd(kernel_size=template.shape)

        dimensions = len(template.shape)
        padding = tuple(side//2 for side in template.shape)

        # Setup convolution to compute correlation to normalized template.
        conv_module = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)[dimensions - 1]
        self._corr = conv_module(1, 1, kernel_size=template.shape, padding=padding, bias=False)
        _normalized_template = (template - template.mean()) / template.std(unbiased=False)
        self._corr.weight.data[0, 0].copy_(_normalized_template / _normalized_template.numel())

    def forward(self, image):
        result = self._corr(image) / self._std(image)

        return result


def test_ncc_nd(template, tolerance=1e-6):
    """
    Tests that values of NCC are correct.

    In particular, applies NCC from a template to itself, and ensures that:
        - Center element is 1.
        - All elements are within the [-1, 1] range.
    """
    ncc_nd = NCC(template)
    result_nd = ncc_nd(template[None, None, ...])

    central_index = tuple(slice(side//2, side//2 + 1) for side in result_nd.shape)
    assert result_nd[central_index].allclose(torch.Tensor((1,))), "Center element is not close to 1."

    assert torch.all(result_nd**2 - tolerance < 1), "Some elements are either below -1 or above 1."

    print(f"All tests passed for {len(template.shape)}d NCC.")


def test_NCC():
    """
    Ensures NCC works as it should for 1d, 2d, and 3d template matching.
    """
    list(map(test_ncc_nd, (torch.randn(d * [5]) for d in (1, 2, 3))))


if __name__ == "__main__":
    test_NCC()

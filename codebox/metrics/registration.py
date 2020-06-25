import SimpleITK as sitk
import numpy as np


def negative_Jacobian(deformation_array, proportion=False):
    """
    calculate the negative Jacobian of the deformation filed
    only for 3D deformation now.
    ==========================================================
    input:
    deformation_array: ndarray of 3D deformation filed
    proportion:

    output:
    cnt: number of negative Jacobin voxels
    cnt_p: proportion of the negative Jacobin voxels to whole image

    """
    if len(deformation_array.shape) != 4:
        raise RuntimeError("Expected dimension 4 but received {}".format(len(deformation_array.shape)))

    w, h, d, n = deformation_array.shape

    if n != 3:
        raise ValueError("Expected last input as (w, h, d, 3) but the last dimension is {}".format(n))

    deformation = sitk.GetImageFromArray(deformation_array)
    determinant = sitk.DisplacementFieldJacobianDeterminant(deformation)
    neg_jacob = (sitk.GetArrayFromImage(determinant)) < 0
    cnt = np.sum(neg_jacob)
    cnt_p = cnt / (w*h*d)

    if proportion:
        return cnt, cnt_p
    else:
        return cnt

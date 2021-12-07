import itk

# TODO: Give credit here
def extract_slice(img, slice_, axis=2, PixelType=itk.F):
    """
    Returns a 2D frame from a 3D image (default assumes itk.F pixels)

    Adds frame to the index of the largest possible region according to the selected axis.  Direction submatrix is maintained
    in returned frame image.

    Parameters
    ----------
    img : 3D ITK image
    slice_ : the frame/slice to get
    axis : a binary array of size dim specifing which index to slice along, the default assumes typical z-axis index
    PixelType : type of pixel in the input and output image. Defaults to floats.
    """

    region = img.GetLargestPossibleRegion()
    size = list(region.GetSize())
    size[axis] = 0
    region.SetSize(size)
    index = list(region.GetIndex())
    index[axis] += slice_
    region.SetIndex(index)
    extractor = itk.ExtractImageFilter[
        itk.Image[PixelType, 3], itk.Image[PixelType, 2]
    ].New(Input=img, ExtractionRegion=region)
    extractor.SetDirectionCollapseToSubmatrix()
    extractor.Update()
    return extractor.GetOutput()


def image_from_array(
    array, spacing=None, direction=None, origin=None, reference_image=None, ttype=None
):
    """
    Augment ITK image_from_array so common parameters can be set in one line
    """
    if reference_image is not None:
        spacing = reference_image.GetSpacing()
        direction = reference_image.GetDirection()
        origin = reference_image.GetOrigin()

    img = itk.image_from_array(array, ttype=ttype)
    if spacing is not None:
        img.SetSpacing(spacing)
    if direction is not None:
        img.SetDirection(direction)
    if origin is not None:
        img.SetOrigin(origin)

    return img

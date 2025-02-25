"""
Segmentation and region properties of images.

Uses cupy and cucim for processing on the GPU

Frames are stacked horizontally on the GPU, to use the GPU memory efficiently.
"""
from skimage.measure import _moments
import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDevice()
except Exception:
    print("Regionprops: Cupy/GPU not available, falling back to CPU")
    cp = None

if cp:
    xp = cp
    from cupyx.scipy.ndimage import label

    STRUCTURE = np.array([[False,  True, False],
                          [ True,  True,  True],
                          [False,  True, False]])

    # Bincount kernel that ignores the background label
    _no_zero_bincount_kernel = cp.ElementwiseKernel(
        'S x', 'raw U bin',
        '''
        if (x > 0)
            atomicAdd(&bin[x], U(1));
        ''',
        'no_zero_bincount_kernel')
else:
    xp = np
    from skimage.measure._label import label



def find_objects(labels):
    """
    Find objects in an image based on labeled regions.

    Parameters
    ----------
    labels: ndarray
        Array containing labeled regions.

    Returns
    -------
    slices: list
        List of slices representing the bounding boxes of each object.
    unique_labels: ndarray
        Array containing the unique labels of the objects.
    """
    non_zero_indices = xp.nonzero(labels)
    values = labels[non_zero_indices]
    unique_labels = xp.unique(values)

    # Try keeping everything on the GPU as long as possible
    label_matches = values[None, :] == unique_labels[:, None]
    # FIXME: this weird type cast is needed to work around a cupy compilation bug on Windows
    if cp:
        p = cp.hstack([0, cp.cumsum(cp.sum(label_matches.astype(cp.float32), axis=1).astype(cp.int32))])
    else:
        p = np.hstack([0, np.cumsum(np.sum(label_matches, axis=1))])
    r = xp.broadcast_to(non_zero_indices[0], label_matches.shape)[label_matches]
    c = xp.broadcast_to(non_zero_indices[1], label_matches.shape)[label_matches]

    if cp:
        objects = p[1:-1].get()
    else:
        objects = p[1:-1]
    r_split = xp.split(r, objects)
    c_split = xp.split(c, objects)
    if r_split[0].size == 0:
        return []
    slices = [
        (
            slice(r_split_element.min().item(), r_split_element.max().item() + 1),
            slice(c_split_element.min().item(), c_split_element.max().item() + 1),
        )
        for r_split_element, c_split_element in zip(r_split, c_split)
    ]
    return slices

def determine_labels(buffer, min_area_pixels, max_area_pixels):
    """
    Determine labels for connected components in a binary image based on their area.

    Parameters
    ----------
    buffer : cp.ndarray or np.ndarray
        The input binary image.
    min_area_pixels : int
        The minimum area (in pixels) for a connected component to be considered valid.
    max_area_pixels : int
        The maximum area (in pixels) for a connected component to be considered valid.
    """
    # binarize the image
    binary_image = (buffer.max() - buffer) > 0
    # label the connected components    

    if cp:
        labels = cp.empty_like(binary_image, dtype=cp.int32)
        label(binary_image, STRUCTURE, output=labels) 
    else:
        labels = label(binary_image, connectivity=1)

    # filter out connected components based on their area    
    if cp:
        component_sizes = xp.zeros(int(labels.max()) + 1, dtype=xp.intp)
        _no_zero_bincount_kernel(labels.ravel(), component_sizes)
    else:
        component_sizes = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    wrong_size = (component_sizes < min_area_pixels) | (component_sizes > max_area_pixels)
    wrong_size_mask = wrong_size[labels]
    labels[wrong_size_mask] = 0
    return labels

def determine_images(labels):
    objects = find_objects(labels)
    if cp:
        images = [(labels[sl] != 0).get() for sl in objects]
    else:
        images = [(labels[sl] != 0) for sl in objects]
    indices_all = [np.nonzero(image) for image in images]
    return images, objects, indices_all

def determine_properties(labels, start_frame_idx, regionprops=("bbox", "ellipse")):
    """
    Determine region properties of connected components in a labeled image.

    Parameters
    ----------
    labels : cp.ndarray or np.ndarray
        The labeled image.
    start_frame_idx : int
        The index of the first frame in the image sequence.
    frame_width : int
        The width of a frame in pixels.
    regionprops : tuple
        A tuple containing the region properties to calculate. Supported values are 'bbox', 'ellipse', and 'orientation'.
        The centroid is always calculated.
    
    Returns
    -------
    properties : dict
        A dictionary containing the requested region properties. Centroid and frame index are always included.
        The keys follow the scikit-image convention for multi-dimensional properties, e.g. the centroid is stored
        as 'centroid-0' and 'centroid-1'.
    """
    images, objects, indices_all = determine_images(labels)

    properties = extract_properties(start_frame_idx, images, objects, indices_all, regionprops)

    return properties

def extract_properties(start_frame_idx, images, objects, indices_all, regionprops=("bbox", "ellipse", "orientation")):
    if "orientation" in regionprops and "ellipse" not in regionprops:
        raise ValueError("The 'orientation' property can only be calculated if 'ellipse' is requested.")
    if not len(indices_all):
        results = {
            "frame": np.array([], dtype=int),
            "centroid-0": np.array([], dtype=float),
            "centroid-1": np.array([], dtype=float),
        }
        if "bbox" in regionprops:
            results["bbox-0"] = np.array([], dtype=int)
            results["bbox-1"] = np.array([], dtype=int)
            results["bbox-2"] = np.array([], dtype=int)
            results["bbox-3"] = np.array([], dtype=int)
        if "ellipse" in regionprops:
            if 'orientation' in regionprops:
                results["orientation"] = np.array([], dtype=float)
            results["minor_axis_length"] = np.array([], dtype=float)
            results["major_axis_length"] = np.array([], dtype=float)
        return results
    

    if "ellipse" not in regionprops:  # For ellipses, we already create the moments
        centroid = [
            np.vstack(
                [
                    indices[0] + sl[0].start,
                    indices[1] + sl[1].start,
                ]
            ).mean(axis=1)
            for indices, sl in zip(indices_all, objects)
        ]

    if "bbox" in regionprops:
        bboxes = [np.array([sl[0].start, sl[1].start, sl[0].stop, sl[1].stop], dtype=np.int32) for sl in objects]

    if "ellipse" in regionprops:
        M_all = [
            _moments.moments(image, 3) for image in images
        ]
        
        centroid_local = [M[tuple(np.eye(2, dtype=int))] / M[(0,) * 2] for M in M_all]
        centroid = [np.array([sl[0].start, sl[1].start]) + c for sl, c in zip(objects, centroid_local)]

        mu_all = [
            _moments.moments_central(
                image,
                c,
                order=2,
            )
            for image, c in zip(images, centroid_local)
        ]

        inertia_tensor_all = [
            _moments.inertia_tensor(image, mu)
            for image, mu in zip(images, mu_all)
        ]
        inertia_tensor_eigvals_all = [
            _moments.inertia_tensor_eigvals(image, T=inertia_tensor)
            for image, inertia_tensor in zip(images, inertia_tensor_all)
        ]

        inertia_tensor_eigvals_all = np.stack(inertia_tensor_eigvals_all)
        major_axis_length = 4 * np.sqrt(inertia_tensor_eigvals_all[:, 0])
        minor_axis_length = 4 * np.sqrt(inertia_tensor_eigvals_all[:, -1])
        
        if 'orientation' in regionprops:
            inertia_tensor_all = np.stack(inertia_tensor_all).reshape(-1, 4)
            a, b, b, c = inertia_tensor_all.T
            orientation = np.where(a - c == 0, np.where(b < 0, -np.pi/4, np.pi/4), np.arctan2(2 * b, a - c) / 2)
    
    properties = {}

    # Centroid and frame information is always stored
    centroid = np.stack(centroid)
    properties["frame"] = np.array([start_frame_idx for _ in objects])
    properties["centroid-0"] = centroid[:, 0]
    properties["centroid-1"] = centroid[:, 1]

    # Bounding boxes
    if "bbox" in regionprops:
        bboxes = np.stack(bboxes)
        # Does it make sense to store bounding boxes in µm instead of pixels ?
        # Here this is mostly for visualization with napari
        properties["bbox-0"] = bboxes[:, 0]
        properties["bbox-1"] = bboxes[:, 1]
        properties["bbox-2"] = bboxes[:, 2]
        properties["bbox-3"] = bboxes[:, 3]

    # Ellipse information (orientation and two axes)
    if "ellipse" in regionprops:
        properties["minor_axis_length"] = minor_axis_length
        properties["major_axis_length"] = major_axis_length
        if 'orientation' in regionprops:
            properties["orientation"] = orientation
    return properties

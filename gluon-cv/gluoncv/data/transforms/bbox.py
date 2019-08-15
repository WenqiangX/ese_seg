"""Bounding boxes transformation functions."""
from __future__ import division
import numpy as np

__all__ = ['crop', 'flip', 'resize', 'val_resize', 'translate']

def crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.

    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox

def flip(bbox, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : type
        Whether flip vertically.

    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    bbox = bbox.copy()
    if flip_y:
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        coef_center_y = height - bbox[:, 5]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
        bbox[:, 5] = coef_center_y
    if flip_x:
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        coef_center_x = width - bbox[:, 4]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
        bbox[:, 4] = coef_center_x
    return bbox

def resize(bbox, in_size, out_size):
    """Resize bouding boxes and coef center according to image resize operation.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 6+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max}, x_{coef_center}, y_{coef_center})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 5] = y_scale * bbox[:, 5]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 4] = x_scale * bbox[:, 4]
    
    return bbox

def val_resize(bbox, in_size, out_size):
    """In val dataset we has 360 polygon points. Resize bouding boxes and polygon points according to image resize operation.
       Val Dataset

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max}, x_{polygon}360, y_{polygon}360)`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]
 
    # bbox and polygon
    # bbox: [0,3] bounding box, [4, 364] polygon_x, [364, 724] polygon_y 
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    
    # resize polygon
    for i in range(360):
        bbox[:, 4+360+i] = y_scale * bbox[:, 4+360+i]
        bbox[:, 4+i] = x_scale * bbox[:, 4+i]
    
    return bbox


def translate(bbox, x_offset=0, y_offset=0):
    """Translate bounding boxes by offsets when training.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 6+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max}, x_{coef_center}, y_{coef_center})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    x_offset : int or float
        Offset along x axis.
    y_offset : int or float
        Offset along y axis.

    Returns
    -------
    numpy.ndarray
        Translated bounding boxes with original shape.
    """
    bbox = bbox.copy()
    bbox[:, :2] += (x_offset, y_offset)
    bbox[:, 2:4] += (x_offset, y_offset)
    bbox[:, 4:6] += (x_offset, y_offset)
    return bbox

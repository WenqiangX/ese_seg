"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import random
import mxnet as mx
from .image import plot_image
import numpy as np
import cv2 as cv
import numpy.polynomial.chebyshev as chebyshev

def cheby(coef):
    """
    coef numpy.array with shape (N , 2*deg+2) such as (N,18), (N,26)
    theta nuumpy.array with shape (360,)    [-1,1]

    Return numpy.array object shape with r (N,360)
    """
    theta = np.linspace(-1, 1, 360)
    coef = coef.T
    r = chebyshev.chebval(theta, coef)

    return r

def plot_r_polygon(img, bboxes,abpoints, coefs, img_w, img_h, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True, deg=8):
    """Visualize bounding boxes and Object Mask ( Object shape ).

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    abpoints : numpy.ndarray or mxnet.nd.NDarray
        shape N,2
    coef : shape N , 2*deg+2
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(abpoints, mx.nd.NDArray):
        abpoints = abpoints.asnumpy()
    if isinstance(coefs, mx.nd.NDArray):
        coefs = coefs.asnumpy()
    if isinstance(img_w, mx.nd.NDArray):
        img_w = img_w.asnumpy()
    if isinstance(img_h, mx.nd.NDArray):
        img_h = img_h.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')
        # demo
        coef = coefs[i].reshape(1,2*deg+2)
        r_all = cheby(coef)  # (1,360)
        bboxw = xmax-xmin
        bboxh = ymax - ymin 
        r_all_real = r_all * np.sqrt(bboxw*bboxw+bboxh*bboxh)
        r_all_real = r_all_real.astype(np.float32).reshape(360,)
        theta_list = np.arange(359 , -1 ,-1)
        theta_list = theta_list.astype(np.float32)
        x, y = cv.polarToCart(r_all_real, theta_list, angleInDegrees=True)
        x = x + float(abpoints[i][0])
        y = y + float(abpoints[i][1])
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)
        polygon = [[int(x[j]), int(y[j])] for j in range(360)]
        polygon = np.array(polygon).reshape((360, 2))
        pgon = plt.Polygon(polygon, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(pgon)
    return ax

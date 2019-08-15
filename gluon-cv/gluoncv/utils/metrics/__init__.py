"""Custom evaluation metrics"""
from __future__ import absolute_import

from .coco_detection import COCODetectionMetric
from .voc_detection import VOCMApMetric, VOC07MApMetric
from .voc_polygon_detection import VOCPolygonMApMetric, VOC07PolygonMApMetric
from .segmentation import SegmentationMetric

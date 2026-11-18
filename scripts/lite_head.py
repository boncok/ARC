# -*- coding: utf-8 -*-
"""Capacity-rebalanced Detect head.

Two variants of the box/cls branch widths are supported:
  * plain 3x3 convs (original liteA head)
  * depthwise-separable branches (box_dw=True) which cost ~4x less per channel and
    free capacity for the backbone/neck.
"""
import copy

import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class DetectLite(Detect):
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(), box_ch=32, cls_ch=64, box_dw=False):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if isinstance(box_ch, (list, tuple)):
            box_ch, cls_ch = int(box_ch[0]), int(box_ch[1])
        self.box_ch, self.cls_ch, self.box_dw = int(box_ch), int(cls_ch), bool(box_dw)
        if self.box_dw:
            self.cv2 = nn.ModuleList(
                nn.Sequential(
                    DWConv(x, x, 3), Conv(x, self.box_ch, 1),
                    DWConv(self.box_ch, self.box_ch, 3), Conv(self.box_ch, self.box_ch, 1),
                    nn.Conv2d(self.box_ch, 4 * self.reg_max, 1),
                )
                for x in ch
            )
        else:
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, self.box_ch, 3), Conv(self.box_ch, self.box_ch, 3), nn.Conv2d(self.box_ch, 4 * self.reg_max, 1))
                for x in ch
            )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, self.cls_ch, 1)),
                nn.Sequential(DWConv(self.cls_ch, self.cls_ch, 3), Conv(self.cls_ch, self.cls_ch, 1)),
                nn.Conv2d(self.cls_ch, self.nc, 1),
            )
            for x in ch
        )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)


def patch_parse_model(box_ch=24, cls_ch=48, box_dw=False):
    """Make ultralytics parse_model build DetectLite whenever the yaml asks for Detect."""
    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "_lite_patched", False):
        return
    orig = tasks.parse_model

    def parse_model(d, ch, verbose=True):
        model, save = orig(d, ch, verbose)
        last = model[-1]
        if isinstance(last, Detect) and not isinstance(last, DetectLite):
            chs = [x[0].conv.in_channels for x in last.cv2]
            lite = DetectLite(nc=last.nc, reg_max=last.reg_max, end2end=False, ch=chs,
                              box_ch=box_ch, cls_ch=cls_ch, box_dw=box_dw)
            lite.i, lite.f = last.i, last.f
            lite.stride = last.stride
            model[-1] = lite
        return model, save

    tasks.parse_model = parse_model
    tasks._lite_patched = True


def register():
    import ultralytics.nn.tasks as tasks

    tasks.DetectLite = DetectLite
    return DetectLite

"""Patch ultralytics NMS to run on CPU.

The Intel Arc (XPU) backend has no fast NMS kernel; TorchNMS falls back to slow
device-side loops (~1.3 s/image at 1024px), which dominates validation time.
NMS on <=300 boxes is microseconds on CPU, so we compute there and move the
kept indices back to the original device.
"""
import torch
import ultralytics.utils.nms as un


_orig_nms = un.TorchNMS.__dict__["nms"].__func__


def _nms_cpu(boxes, scores, iou_threshold):
    device = boxes.device
    if device.type == "xpu":
        idx = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
        return idx.to(device)
    return _orig_nms(boxes, scores, iou_threshold)


un.TorchNMS.nms = staticmethod(_nms_cpu)
print("[xpu_patch] NMS patched to CPU")

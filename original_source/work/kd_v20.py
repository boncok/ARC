# -*- coding: utf-8 -*-
"""v20 distillation: kd_aligned2 (feature + class-logit KD at P3/P4) PLUS a
channel-normalised P2 feature term.

Why: the honest-holdout error analysis shows the student's deficit is classification
calibration, and ~76% of its anchors live at P2 — a level the teacher has no head for,
so kd_aligned2 leaves P2 completely unsupervised. kd_aligned3 tried raw MSE there and
diverged (256ch x 80x80 map vs detection loss ~16). Normalising both maps over the
channel axis bounds the term in [0, 2], so dis_p2 can be raised safely.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kd_aligned import FeatureHook
from kd_aligned2 import KdModel2


class KdV20(KdModel2):
    def __init__(self, teacher_model, student_model, offset: int = 1, dis: float | None = None,
                 dis_cls: float = 1.0, dis_p2: float = 0.5, t_p2_idx: int = 2):
        self.dis_p2 = float(dis_p2)
        self.t_p2_idx = int(t_p2_idx)
        self._p2_feats: dict = {}
        self._p2_hooks: list = []
        super().__init__(teacher_model, student_model, offset=offset, dis=dis, dis_cls=dis_cls)
        self._p2_s_idx = self.get_distill_layers(self.student_model)[0]
        self._register_p2_hooks()

        device = next(self.student_model.parameters()).device
        ch = self.student_model.yaml.get("channels", 3)
        _args = getattr(self.student_model, "args", {})
        imgsz = _args["imgsz"] if isinstance(_args, dict) else _args.imgsz
        self.student_model.eval()
        with torch.no_grad():
            im = torch.zeros(1, ch, imgsz, imgsz, device=device)
            self.teacher_model(im)
            self.student_model(im)
        self.student_model.train()
        s, t = self._p2_feats[0], self._p2_feats[1]
        assert s.shape[-2:] == t.shape[-2:], f"P2 grid mismatch {s.shape} vs {t.shape}"
        self.p2_proj = nn.Sequential(
            nn.Conv2d(s.shape[1], t.shape[1], 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(t.shape[1], t.shape[1], 1, 1, 0),
        ).to(device)
        print(f"[kd-v20] P2 cosine KD student_l={self._p2_s_idx} teacher_l={self.t_p2_idx} "
              f"ch {s.shape[1]}->{t.shape[1]} grid={tuple(s.shape[-2:])} dis_p2={self.dis_p2} "
              f"proj_params={sum(p.numel() for p in self.p2_proj.parameters())}", flush=True)

    def _register_p2_hooks(self):
        for h in self._p2_hooks:
            h.remove()
        self._p2_hooks = [
            self.student_model.model[self._p2_s_idx].register_forward_hook(FeatureHook(self._p2_feats, 0)),
        ]
        if self.teacher_model is not None:
            self._p2_hooks.append(
                self.teacher_model.model[self.t_p2_idx].register_forward_hook(FeatureHook(self._p2_feats, 1)))

    def __getstate__(self):
        state = super().__getstate__()
        state["_p2_hooks"] = []
        state["_p2_feats"] = {}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._p2_feats = {}
        self._register_p2_hooks()

    def loss(self, batch, preds=None):
        out = super().loss(batch, preds)
        if self.training and self.dis_p2 > 0 and 0 in self._p2_feats and 1 in self._p2_feats:
            s = F.normalize(self.p2_proj(self._p2_feats[0].float()), dim=1)
            t = F.normalize(self._p2_feats[1].float().detach(), dim=1)
            term = (1.0 - (s * t).sum(1)).mean() * self.dis_p2
            out[0][-1] = out[0][-1] + term * batch["img"].shape[0]
            out[1]["dis_loss"] = out[1]["dis_loss"] + term.detach()
        return out


class _PatchedV20(KdV20):
    _offset = 1
    _dis = None
    _dis_cls = 1.0
    _dis_p2 = 0.5
    _t_p2_idx = 2

    def __init__(self, teacher_model, student_model):
        super().__init__(teacher_model, student_model, offset=type(self)._offset, dis=type(self)._dis,
                         dis_cls=type(self)._dis_cls, dis_p2=type(self)._dis_p2, t_p2_idx=type(self)._t_p2_idx)


def install(offset: int = 1, dis: float | None = None, dis_cls: float = 1.0,
            dis_p2: float = 0.5, t_p2_idx: int = 2):
    import ultralytics.engine.trainer as tr
    import ultralytics.nn.distill_model as dm

    _PatchedV20.__name__ = "DistillationModel"
    _PatchedV20._offset = offset
    _PatchedV20._dis = dis
    _PatchedV20._dis_cls = dis_cls
    _PatchedV20._dis_p2 = dis_p2
    _PatchedV20._t_p2_idx = t_p2_idx
    dm.DistillationModel = _PatchedV20
    tr.DistillationModel = _PatchedV20
    return _PatchedV20
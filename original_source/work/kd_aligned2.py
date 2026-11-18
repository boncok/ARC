# -*- coding: utf-8 -*-
"""Level-aligned distillation + class-logit KD for the P2-P4 student.

Adds a second term on top of work/kd_aligned.py: at the two shared levels
(student P3/P4 = teacher P3/P4, same stride so identical anchor grid) the
student's class logits are pulled towards the teacher's sigmoid probabilities.
This targets the observed failure mode (class confusion on toy_car/phone/cup),
which feature-only MSE distillation does not address directly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import kd_aligned
from kd_aligned import AlignedDistillationModel


class KdModel2(AlignedDistillationModel):
    def __init__(self, teacher_model, student_model, offset: int = 1, dis: float | None = None,
                 dis_cls: float = 0.5):
        super().__init__(teacher_model, student_model, offset=offset, dis=dis)
        self.dis_cls = float(dis_cls)
        self._dbg = False
        print(f"[kd-aligned2] class-logit KD enabled dis_cls={self.dis_cls}", flush=True)

    @staticmethod
    def _head_dict(head):
        o = head
        if isinstance(o, (tuple, list)):
            o = o[1]
        return o

    @classmethod
    def _split_levels(cls, head):
        """Split the head class-score tensor per pyramid level using its own 'feats' grids."""
        d = cls._head_dict(head)
        scores = d["scores"]
        areas = [f.shape[-2] * f.shape[-1] for f in d["feats"]]
        if scores.dim() == 3 and scores.shape[-1] == sum(areas):
            return torch.split(scores, areas, dim=-1)
        return list(scores)

    def _cls_kd(self, student_head, teacher_head) -> torch.Tensor:
        s_parts = self._split_levels(student_head)
        t_parts = self._split_levels(teacher_head)
        loss = None
        for i in range(self.n_used):
            sp = s_parts[self.offset + i]
            tp = t_parts[i]
            if sp.shape != tp.shape:
                if sp.numel() != tp.numel():
                    raise RuntimeError(f"cls KD shape mismatch {tuple(sp.shape)} vs {tuple(tp.shape)}")
                sp = sp.reshape(tp.shape)
            term = F.binary_cross_entropy_with_logits(sp, tp.sigmoid().detach())
            if not getattr(self, "_dbg", False):
                self._dbg = True
                print(f"[kd-aligned2] s{tuple(sp.shape)} t{tuple(tp.shape)} bce={float(term):.4f}", flush=True)
            loss = term if loss is None else loss + term
        return loss / max(self.n_used, 1) * self.dis_cls

    def loss(self, batch, preds=None):
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])
        regular_loss, loss_items = self.student_model.loss(batch, preds)

        n_t = len(self._t_all_idx) - 1
        teacher_head = self._teacher_feats[n_t]
        teacher_scores = (
            self.decouple_outputs(teacher_head, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head, branch="one2one")["scores"]
        ) / 2
        all_neck = [self._teacher_feats[i] for i in range(n_t)]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in all_neck], dim=-1)
        scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts[: self.n_used])

        for i in range(self.n_used):
            teacher_feat = self.decouple_outputs(self._teacher_feats[i])
            student_feat = self.projector[i](self.decouple_outputs(self._student_feats[i]))
            loss_distill = loss_distill + self.loss_sl2(student_feat, teacher_feat, i, scores) * self.dis

        if self.dis_cls > 0:
            loss_distill = loss_distill + self._cls_kd(preds, teacher_head)

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items


class _Patched2(KdModel2):
    _offset = 1
    _dis = None
    _dis_cls = 0.5

    def __init__(self, teacher_model, student_model):
        AlignedDistillationModel.__init__(self, teacher_model, student_model,
                                          offset=type(self)._offset, dis=type(self)._dis)
        self.dis_cls = float(type(self)._dis_cls)
        print(f"[kd-aligned2] class-logit KD enabled dis_cls={self.dis_cls}", flush=True)


def install2(offset: int = 1, dis: float | None = None, dis_cls: float = 0.5):
    import ultralytics.engine.trainer as tr
    import ultralytics.nn.distill_model as dm

    _Patched2.__name__ = "DistillationModel"
    _Patched2._offset = offset
    _Patched2._dis = dis
    _Patched2._dis_cls = dis_cls
    dm.DistillationModel = _Patched2
    tr.DistillationModel = _Patched2
    return _Patched2

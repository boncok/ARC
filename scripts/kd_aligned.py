# -*- coding: utf-8 -*-
"""Level-aligned knowledge distillation for a P2-P4 student and a P3-P5 teacher.

The stock ultralytics DistillationModel pairs feature levels by list index, which
mismatches scales when the student has an extra high-resolution level (P2/4):
student P2(4) would be paired with teacher P3(8). This variant pairs
student level (offset + i) with teacher level i and keeps the teacher head hooked
for the score map used to weight the feature loss.

Student levels [P2/4, P3/8, P4/16] + head, teacher levels [P3/8, P4/16, P5/32] + head:
offset=1 -> student P3/P4 are distilled from teacher P3/P4, student P2 has no teacher.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import copy_attr


class FeatureHook:
    """Picklable forward hook storing layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        self.feat_dict[self.idx] = output


class AlignedDistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model, offset: int = 1, dis: float | None = None):
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model

        s_layers = self.get_distill_layers(student_model)  # [neck..., head]
        t_layers = self.get_distill_layers(teacher_model)
        n_s, n_t = len(s_layers) - 1, len(t_layers) - 1
        offset = int(offset)
        if offset < 0 or n_s - offset > n_t:
            raise ValueError(f"offset {offset} invalid for student {n_s} levels / teacher {n_t} levels")
        self.offset = offset
        self.n_used = n_s - offset

        # student: keep neck levels [offset:] plus the head; hooks keyed by (0..n_used-1, n_used)
        self.feats_idx = s_layers[offset:]
        # teacher: hook every neck level (needed to split the head score map) plus the head
        self._t_all_idx = t_layers  # keys 0..n_t-1 neck, n_t head

        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        _args = getattr(student_model, "args", {})
        imgsz = _args["imgsz"] if isinstance(_args, dict) else _args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()

        s_out = [self._student_feats[i] for i in range(self.n_used + 1)]
        t_out = [self._teacher_feats[i] for i in range(n_t + 1)]
        copy_attr(self, student_model)
        if dis is None:
            a = getattr(self.student_model, "args", {})
            dis = a["dis"] if isinstance(a, dict) else a.dis
        self.dis = float(dis)

        projectors = []
        for i in range(self.n_used):
            sd = self.decouple_outputs(s_out[i]).shape[1]
            td = self.decouple_outputs(t_out[i]).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(sd, td, 1, 1, 0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(td, td, 1, 1, 0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

        # sanity: spatial sizes must match at every distilled level
        for i in range(self.n_used):
            ss = tuple(self.decouple_outputs(s_out[i]).shape[-2:])
            ts = tuple(self.decouple_outputs(t_out[i]).shape[-2:])
            assert ss == ts, f"level {i}: student {ss} != teacher {ts}"
        print(
            f"[kd-aligned] offset={offset} student_levels={self.feats_idx} teacher_levels={self._t_all_idx} "
            f"dis={self.dis} student_ch={[self.decouple_outputs(x).shape[1] for x in s_out[:-1]]} "
            f"teacher_ch={[self.decouple_outputs(x).shape[1] for x in t_out[:-1]]}",
            flush=True,
        )

    # ---- plumbing (same as stock) ----
    def __getstate__(self):
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        self._remove_feature_hooks()
        for i, s in enumerate(self.feats_idx):
            self._clear_feature_hooks(self.student_model.model[s])
            self._student_hooks.append(
                self.student_model.model[s].register_forward_hook(FeatureHook(self._student_feats, i))
            )
        if self.teacher_model is not None:
            for i, t in enumerate(self._t_all_idx):
                self._clear_feature_hooks(self.teacher_model.model[t])
                self._teacher_hooks.append(
                    self.teacher_model.model[t].register_forward_hook(FeatureHook(self._teacher_feats, i))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose: bool = True, imgsz=640):
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

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

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items

    @staticmethod
    def loss_sl2(student_feat, teacher_feat, feat_idx, teacher_scores):
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value):
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds


class _Patched(AlignedDistillationModel):
    """Module-level subclass so checkpoint deepcopy/pickle can resolve the class."""

    _offset = 1
    _dis = None

    def __init__(self, teacher_model, student_model):
        super().__init__(teacher_model, student_model, offset=type(self)._offset, dis=type(self)._dis)


def install(offset: int = 1, dis: float | None = None):
    """Monkeypatch ultralytics so the trainer builds the level-aligned wrapper."""
    import ultralytics.engine.trainer as tr
    import ultralytics.nn.distill_model as dm

    _Patched.__name__ = "DistillationModel"
    _Patched._offset = offset
    _Patched._dis = dis
    dm.DistillationModel = _Patched
    tr.DistillationModel = _Patched
    return _Patched

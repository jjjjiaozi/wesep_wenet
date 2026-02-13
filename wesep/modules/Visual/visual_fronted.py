import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 相对导入（确保同目录下有 __init__.py）
from .visual_encoder import VisualEncoder
print("[DBG] importing visual_fronted from:", __file__)


class BaseVisualFeature(nn.Module):
    def compute(self, visual_feat, T, **kwargs):
        raise NotImplementedError

    def post(self, mix_repr, feat_repr, **kwargs):
        raise NotImplementedError


class VisualContextFeature(BaseVisualFeature):
    """
    中层融合：把 visual 融合进 subband_feature
    mix_repr: (B, nband, Fd, T)
    visual_feat:
      - raw:        (B, Tv, Dv)  -> VisualEncoder -> (B, Cv, Tv) -> interpolate -> (B, Cv, T)
      - precomputed:(B, Tv, Cv)  -> transpose    -> (B, Cv, Tv) -> interpolate -> (B, Cv, T)
    """

    def __init__(
        self,
        dv=512,
        cv=256,
        feature_dim=128,
        nband=None,
        pretrained=None,
        freeze=False,
        input_is_precomputed=False,   # ✅ 新增：True 表示 npy 已经是 encoder 特征
        precomputed_dim=None,         # ✅ 若 precomputed 维度不是 cv，可填它并自动投影到 cv
    ):
        super().__init__()
        assert nband is not None, "need nband from sep_model.nband"
        self.nband = nband
        self.feature_dim = feature_dim
        self.cv = cv
        self.input_is_precomputed = input_is_precomputed

        # raw 输入才需要 encoder
        self.vis_encoder = None
        if not input_is_precomputed:
            self.vis_encoder = VisualEncoder(dv=dv, cv=cv, pretrained=pretrained, freeze=freeze)

        # 如果 precomputed_dim != cv，用 1x1 Conv1d 投影到 cv
        self.pre_proj = None
        if input_is_precomputed and (precomputed_dim is not None) and (precomputed_dim != cv):
            self.pre_proj = nn.Conv1d(precomputed_dim, cv, kernel_size=1, bias=False)

        # concat 后用 1x1 Conv1d 融回 feature_dim
        self.fuse = nn.Conv1d(feature_dim + cv, feature_dim, kernel_size=1, bias=False)
        # ✅ 自动推断输入通道数（第一次 forward 时决定 in_channels）
        #self.fuse = nn.LazyConv1d(feature_dim, kernel_size=1, bias=False)

        self.act = nn.PReLU()
        self.norm = nn.BatchNorm1d(feature_dim)
        if not hasattr(self, "_dbg_cfg_once"):
            print("[DBG-VC-init] input_is_precomputed =", input_is_precomputed,
                "cv =", cv,
                "precomputed_dim =", precomputed_dim)
            self._dbg_cfg_once = True


    # def compute(self, visual_feat, T, **kwargs):
    #     """
    #     return: (B, Cv, T)
    #     """
    #     if not hasattr(self, "_dbg_in_once"):
    #         print("[DBG-VC-compute] visual_feat.shape =", visual_feat.shape)
    #         self._dbg_in_once = True

    #             # visual_feat: (B, Tv, D)  -> 我们内部统一变成 (B, C, Tv)
    #     if self.input_is_precomputed:
    #         # visual_feat assumed (B, Tv, Dpre)
    #         vis = visual_feat.transpose(1, 2).contiguous()  # (B, Dpre, Tv)
    #         if self.pre_proj is not None:
    #             vis = self.pre_proj(vis)  # (B, Cv, Tv)
    #     else:
    #         # raw: (B, Tv, Dv) -> encoder -> (B, Cv, Tv)
    #         vis = self.vis_encoder(visual_feat)  # (B, Cv, Tv)

    #     # 对齐到音频时间帧 T
    #     vis = F.interpolate(vis, size=T, mode="linear", align_corners=False)  # (B, Cv, T)
    #     return vis
    def compute(self, visual_feat, T, **kwargs):
        # visual_feat: (B, Tv, Dpre)
        if self.input_is_precomputed:
            vis = visual_feat.transpose(1, 2).contiguous()  # (B, Dpre, Tv)

            # ✅ 强制保证通道变成 cv
            if vis.size(1) != self.cv:
                # 如果 pre_proj 没建，或 in_channels 不匹配，就重建一个正确的
                if (self.pre_proj is None) or (self.pre_proj.in_channels != vis.size(1)) or (self.pre_proj.out_channels != self.cv):
                    self.pre_proj = nn.Conv1d(vis.size(1), self.cv, kernel_size=1, bias=False).to(vis.device)
                vis = self.pre_proj(vis)  # (B, cv, Tv)
        else:
            vis = self.vis_encoder(visual_feat)  # (B, cv, Tv)

        vis = F.interpolate(vis, size=T, mode="linear", align_corners=False)  # (B, cv, T)

        # ✅ 保险：不满足就立刻报出真实维度
        assert vis.size(1) == self.cv, f"visual channel mismatch: got {vis.size(1)} but cv={self.cv}"
        return vis


    def post(self, mix_repr, feat_repr, **kwargs):
        """
        mix_repr : (B, nband, Fd, T)
        feat_repr: (B, Cv, T)
        return   : (B, nband, Fd, T)
        """
        B, nband, Fd, T = mix_repr.shape
        assert nband == self.nband and Fd == self.feature_dim

        # (B, Cv, T) -> (B, nband, Cv, T)
        vis_band = feat_repr.unsqueeze(1).expand(B, nband, feat_repr.size(1), T)

        # concat -> (B, nband, Fd+Cv, T)
        x = torch.cat([mix_repr, vis_band], dim=2)

        # reshape -> Conv1d expects (N, C, T)
        x = x.reshape(B * nband, Fd + feat_repr.size(1), T)
        x = self.fuse(x)
        x = self.act(x)
        x = self.norm(x)

        x = x.reshape(B, nband, Fd, T)
        return x

class VisualFrontend(nn.Module):
    DEFAULT = {
        "features": {
            "viscontext": {
                "enabled": True,
                "dv": 512,
                "cv": 256,
                "pretrained": None,
                "freeze": False,

                # ✅ 新增：你的 npy 已经是 encoder 特征，就设 True
                "input_is_precomputed": False,

                # ✅ 新增：如果 npy 的维度不是 cv，用它来做投影；否则可不填/填同样值
                "precomputed_dim": None,
            },
        }
    }

    def __init__(self, config, feature_dim, nband):
        super().__init__()
        cfg = {**VisualFrontend.DEFAULT, **(config or {})}
        self.cfg = cfg

        self.features = nn.ModuleDict()

        if cfg["features"]["viscontext"]["enabled"]:
            vc = cfg["features"]["viscontext"]
            self.features["viscontext"] = VisualContextFeature(
                dv=vc["dv"],
                cv=vc["cv"],
                feature_dim=feature_dim,
                nband=nband,
                pretrained=vc.get("pretrained", None),
                freeze=vc.get("freeze", False),

                # ✅ 把新增字段传下去
                input_is_precomputed=vc.get("input_is_precomputed", False),
                precomputed_dim=vc.get("precomputed_dim", None),
            )

    def compute_all(self, visual_feat, T, **kwargs):
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(visual_feat, T, **kwargs)
        return out

    def post_all(self, mix_repr, feat_dict, **kwargs):
        x = mix_repr
        for name, module in self.features.items():
            x = module.post(x, feat_dict[name], **kwargs)
        return x

    def forward(self, mix_repr, visual_feat, T=None, **kwargs):
        if T is None:
            T = mix_repr.shape[-1]
        feat_dict = self.compute_all(visual_feat, T, **kwargs)
        out = self.post_all(mix_repr, feat_dict, **kwargs)
        return out, feat_dict

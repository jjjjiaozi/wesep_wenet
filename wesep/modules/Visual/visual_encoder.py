# visual_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    """
    input : visual_feat (B, Tv, Dv)
    output: vis_seq     (B, Cv, Tv)  
    """
    def __init__(self, dv=512, cv=256, use_conv=True, pretrained=None, freeze=False):
        super().__init__()
        self.proj = nn.Linear(dv, cv)
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(cv, cv, kernel_size=3, padding=1, bias=False),
                nn.PReLU(),
                nn.BatchNorm1d(cv),
            )
        if pretrained is not None:
            ckpt = torch.load(pretrained, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            # 去 module.
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.load_state_dict(state, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, visual_feat):
        # (B, Tv, Dv) -> (B, Tv, Cv)
        x = self.proj(visual_feat)
        # -> (B, Cv, Tv)
        x = x.transpose(1, 2).contiguous()
        if self.use_conv:
            x = self.conv(x)
        return x

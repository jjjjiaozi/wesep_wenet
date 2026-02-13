# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update
from wesep.modules.Visual.visual_fronted import VisualFrontend


class TSE_BSRNN_VISUAL(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ===== Merge configs =====
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,    # For Separation (multiple output)
            spec_dim=2 # For TSE feature, used in self.subband_norm
        )
        sep_configs = {**sep_configs, **config["separator"]}

        spk_configs = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": sep_configs["win"],
                    "hop": sep_configs["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": sep_configs["causal"],
                    "enc_dim": sep_configs["win"] // 2 + 1,
                    "emb_dim": sep_configs["feature_dim"] // 2,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"],
                    "atten_dim": sep_configs["feature_dim"]
                },
                "spkemb": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"]
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": sep_configs["sr"]
                },
            },
        }
        self.spk_configs = deep_update(spk_configs, config.get("speaker", {}))

        # ===== Separator Loading =====
        # 注意：usef/tfmap 会改变 spec_dim（即 spec_RI 的通道数），这是“结构级别”的改变
        if self.spk_configs["features"]["usef"]["enabled"]:
            sep_configs["spec_dim"] = self.spk_configs["features"]["usef"]["emb_dim"] * 2
        if self.spk_configs["features"]["tfmap"]["enabled"]:
            sep_configs["spec_dim"] = sep_configs["spec_dim"] + 1

        self.sep_model = BSRNN(**sep_configs)

        # ===== Speaker Loading =====
        if self.spk_configs["features"]["context"]["enabled"]:
            self.spk_configs["features"]["context"]["band"] = self.sep_model.nband
        self.spk_ft = SpeakerFrontend(self.spk_configs)

        # ===== Visual Loading (optional) =====
        self.visual_cfg = config.get("visual", None)
        self.visual_ft = None
        if self.visual_cfg is not None:
            self.visual_ft = VisualFrontend(
                config=self.visual_cfg,
                feature_dim=sep_configs["feature_dim"],
                nband=self.sep_model.nband
            )

    def forward(self, mix, enroll=None, visual_feat=None):
        """
        mix:        (B, T)
        enroll:     (B, Te) or None
        visual_feat:(B, Tv, Dv) or None
        """

        # ---- basic checks ----
        assert mix.dim() == 2, "Only support 2D Input: mix should be (B, T)"

        if enroll is None and visual_feat is None:
            raise ValueError("Need at least one cue: enroll or visual_feat.")

        # 当 enroll=None 时，listen/usef/tfmap 这些必须禁用，否则 spec_dim 结构不匹配
        if enroll is None:
            if self.spk_configs["features"]["listen"]["enabled"] or \
               self.spk_configs["features"]["usef"]["enabled"] or \
               self.spk_configs["features"]["tfmap"]["enabled"]:
                raise ValueError(
                    "enroll is None but listen/usef/tfmap is enabled in config. "
                    "These features require enroll and also change spec_dim. "
                    "Please disable them when running visual-only mode."
                )

        wav_mix = mix
        wav_enroll = enroll  # may be None

        ###########################################################
        # C0. Feature: listen (requires enroll)
        if (wav_enroll is not None) and self.spk_configs["features"]["listen"]["enabled"]:
            # Prepend the enroll to the mix in the beginning
            wav_mix = self.spk_ft.listen.compute(wav_enroll, wav_mix)  # (B, T_e + T_s + T)

        ###########################################################
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(wav_mix)[-1]  # (B, F, T) complex
        # S2. Concat real and imag
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # (B, 2, F, T)

        ###########################################################
        # C1. Feature: usef (requires enroll, changes spec_dim)
        if (wav_enroll is not None) and self.spk_configs["features"]["usef"]["enabled"]:
            enroll_spec = self.sep_model.stft(wav_enroll)[-1]  # (B, F, T_e) complex
            enroll_spec = torch.stack([enroll_spec.real, enroll_spec.imag], 1)  # (B, 2, F, T_e)

            enroll_usef, mix_usef = self.spk_ft.usef.compute(enroll_spec, spec_RI)  # (B, embed_dim, F, T)
            spec_RI = self.spk_ft.usef.post(mix_usef, enroll_usef)  # (B, embed_dim*2, F, T)

        # C2. Feature: tfmap (requires enroll, changes spec_dim)
        if (wav_enroll is not None) and self.spk_configs["features"]["tfmap"]["enabled"]:
            enroll_mag = self.sep_model.stft(wav_enroll)[0]  # (B, F, T_e)
            enroll_tfmap = self.spk_ft.tfmap.compute(enroll_mag, torch.abs(spec))  # (B, F, T)
            spec_RI = self.spk_ft.tfmap.post(spec_RI, enroll_tfmap.unsqueeze(1))  # (B, 3 or ..., F, T)

        ###########################################################
        # Split to subbands

        subband_spec = self.sep_model.band_split(spec_RI)  # list of (B, spec_dim, BW, T)

        # if not hasattr(self, "_dbg_once2"):
        #     print("[DBG] subband_spec[0] shape:", subband_spec[0].shape)  # (B, C, BW, T)

        subband_mix_spec = self.sep_model.band_split(spec)  # list of (B, BW, T) complex

        # S3. Normalization and bottleneck
        subband_feature = self.sep_model.subband_norm(subband_spec)  # (B, nband, feat, T)

        # if not hasattr(self, "_dbg_once3"):
        #     print("[DBG] subband_feature shape:", subband_feature.shape)   # (B, nband, Fd, T)
        #     self._dbg_once = self._dbg_once2 = self._dbg_once3 = True


        ###########################################################
        # V1. Visual cue (optional, does NOT change spec_dim)
        # if (self.visual_ft is not None) and (visual_feat is not None):
        #     # Expect visual_feat: (B, Tv, Dv)
        #     subband_feature, _ = self.visual_ft(subband_feature, visual_feat)
        if (self.visual_ft is not None) and (visual_feat is not None):
            # visual_feat: (B, Tv, Dv) OR (B, Tv, Cv) if precomputed
            subband_feature, _ = self.visual_ft(subband_feature, visual_feat, T=subband_feature.size(-1))

        ###########################################################
        # C3. Feature: context (requires enroll)
        if (wav_enroll is not None) and self.spk_configs["features"]["context"]["enabled"]:
            enroll_context = self.spk_ft.context.compute(wav_enroll)  # (B, F_e, T_e)
            subband_feature = self.spk_ft.context.post(subband_feature, enroll_context)  # (B, nband, feat, T)

        # C4. Feature: spkemb (requires enroll)
        if (wav_enroll is not None) and self.spk_configs["features"]["spkemb"]["enabled"]:
            enroll_emb = self.spk_ft.spkemb.compute(wav_enroll)  # (B, F_e)
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)    # (B, 1, F_e, 1)
            subband_feature = self.spk_ft.spkemb.post(subband_feature, enroll_emb)  # (B, nband, feat, T)

        ###########################################################
        # S4. Separation
        sep_output = self.sep_model.separator(subband_feature)  # (B, nband, feat, T)

        # S5. Complex Mask
        est_spec_RI = self.sep_model.band_masker(sep_output, subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0], est_spec_RI[:, 1])       # (B, S, F, T)

        # S6. Back into waveform
        output = self.sep_model.istft(est_complex)  # (B, S, T)
        s = torch.squeeze(output, dim=1)            # (B, T)

        ###########################################################
        # C0. Feature: listen post (requires enroll)
        if (wav_enroll is not None) and self.spk_configs["features"]["listen"]["enabled"]:
            s = self.spk_ft.listen.post(s)  # (B, T)

        return s


def check_causal(model):
    input = torch.randn(1, 16000 * 8).clamp_(-1, 1)
    enroll = torch.randn(1, 16000 * 2).clamp_(-1, 1)
    fs = 16000
    model = model.eval()
    with torch.no_grad():
        out1 = model(input, enroll=enroll, visual_feat=None)
        for i in range(fs * 1, fs * 4, fs):
            inputs2 = input.clone()
            inputs2[..., i:] = 1 + torch.rand_like(inputs2[..., i:])
            out2 = model(inputs2, enroll=enroll, visual_feat=None)
            print((((out1[0] - out2[0]).abs() > 1e-8).float().argmax()) / fs)
            print((((inputs2 - input).abs() > 1e-8).float().argmax()) / fs)


if __name__ == "__main__":
    config = dict()
    config["separator"] = dict(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        causal=True,
        nspk=1,
    )
    config["speaker"] = {
        "features": {
            "listen": {"enabled": True},
            "usef": {"enabled": True},
            "tfmap": {"enabled": True},
            "context": {"enabled": True},
            "spkemb": {"enabled": True},
        }
    }
    # ✅ visual config example (enable when you have visual_feat)
    config["visual"] = {
        "features": {
            "viscontext": {
                "enabled": True,
                "dv": 512,
                "cv": 256,
                "pretrained": None,
                "freeze": False
            }
        }
    }

    model = TSE_BSRNN_SPK(config)
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))

    mix = torch.randn(4, 32000)
    enroll = torch.randn(4, 31235)
    visual_feat = torch.randn(4, 75, 512)  # example: (B, Tv, Dv)

    model = model.eval()
    with torch.no_grad():
        # enroll-only
        out1 = model(mix, enroll=enroll, visual_feat=None)
        print("enroll-only:", out1.shape)

        # visual-only (IMPORTANT: listen/usef/tfmap must be disabled in config)
        # out2 = model(mix, enroll=None, visual_feat=visual_feat)
        # print("visual-only:", out2.shape)

        # enroll + visual
        out3 = model(mix, enroll=enroll, visual_feat=visual_feat)
        print("enroll+visual:", out3.shape)

    check_causal(model)

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Device ──────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

# ── Checkpoint paths ─────────────────────────────────────────

BODY_CKPT  = Path("checkpoints_cvae_pl_pt")
LAYER_CKPT = Path("checkpoints_cvae_layer_pt")

BODY_ACTIONS = [
    "walk", "idle", "run", "slash", "shoot",
    "thrust", "jump", "sit", "spellcast", "other"
]

LAYER_ORDER = ["body", "legs", "feet", "torso", "hair"]

CATEGORY_KR = {
    "body":  "몸",
    "hair":  "헤어",
    "torso": "상의",
    "legs":  "하의",
    "feet":  "발",
}

# ── Model architecture ───────────────────────────────────────

class CVAEEncoder(nn.Module):
    def __init__(self, num_classes, base_ch=32, latent_dim=128, channels=4):
        super().__init__()
        layers, in_ch, ch = [], channels, base_ch
        for _ in range(4):
            layers += [nn.Conv2d(in_ch, ch, 4, stride=2, padding=1),
                       nn.BatchNorm2d(ch, momentum=0.1),
                       nn.LeakyReLU(0.2)]
            in_ch = ch; ch *= 2
        self.conv  = nn.Sequential(*layers)
        flat_dim   = (base_ch * 8) * 4 * 4
        self.fc_mu = nn.Linear(flat_dim + num_classes, latent_dim)
        self.fc_lv = nn.Linear(flat_dim + num_classes, latent_dim)

    def forward(self, x, c):
        h = self.conv(x).flatten(1)
        return self.fc_mu(torch.cat([h, c], -1)), self.fc_lv(torch.cat([h, c], -1))


class CVAEDecoder(nn.Module):
    def __init__(self, num_classes, base_ch=32, latent_dim=128, channels=4):
        super().__init__()
        self.start_ch = base_ch * 8
        self.fc = nn.Linear(latent_dim + num_classes, self.start_ch * 4 * 4)
        layers, in_ch = [], self.start_ch
        for i, out_ch in enumerate([base_ch*4, base_ch*2, base_ch, channels]):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            if i < 3:
                layers += [nn.BatchNorm2d(out_ch, momentum=0.1), nn.ReLU()]
            else:
                layers.append(nn.Sigmoid())
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], -1))
        return self.deconv(h.view(-1, self.start_ch, 4, 4))


class CVAE(nn.Module):
    def __init__(self, num_classes, base_ch=32, latent_dim=128, channels=4):
        super().__init__()
        self.encoder = CVAEEncoder(num_classes, base_ch, latent_dim, channels)
        self.decoder = CVAEDecoder(num_classes, base_ch, latent_dim, channels)

    def forward(self, x, c):
        mu, lv = self.encoder(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decoder(z, c), mu, lv


# ── Model loading ────────────────────────────────────────────

@st.cache_resource
def load_body_model():
    cfg = json.loads((BODY_CKPT / "config.json").read_text())
    model = CVAE(cfg["num_classes"], cfg["base_channels"], cfg["latent_dim"], cfg["channels"])
    model.load_state_dict(torch.load(BODY_CKPT / "best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, cfg


@st.cache_resource
def load_layer_model():
    cfg = json.loads((LAYER_CKPT / "config.json").read_text())
    model = CVAE(cfg["num_classes"], cfg["base_channels"], cfg["latent_dim"], cfg["channels"])
    model.load_state_dict(torch.load(LAYER_CKPT / "best.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, cfg


# ── Generation ───────────────────────────────────────────────

def generate_layer(model, label_idx: int, num_classes: int, latent_dim: int) -> np.ndarray:
    z = torch.randn(1, latent_dim).to(DEVICE)
    c = F.one_hot(torch.tensor([label_idx]), num_classes).float().to(DEVICE)
    with torch.no_grad():
        img = model.decoder(z, c)
    return img[0].cpu().permute(1, 2, 0).numpy()


def composite(arrays: list[np.ndarray]) -> Image.Image:
    base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for arr in arrays:
        layer = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), "RGBA")
        base = Image.alpha_composite(base, layer)
    return base


def arr_to_pil(arr: np.ndarray, size: int = 128) -> Image.Image:
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.NEAREST)


# ── App ──────────────────────────────────────────────────────

st.set_page_config(page_title="LPC Sprite Generator", layout="wide")
st.title("LPC Sprite Generator")

body_model, body_cfg   = load_body_model()
layer_model, layer_cfg = load_layer_model()

label_map   = layer_cfg["label_map"]
label_names = layer_cfg["label_names"]


def keywords(cat: str) -> list[str]:
    prefix = cat + "_"
    return sorted(n[len(prefix):] for n in label_names if n.startswith(prefix))


with st.sidebar:
    st.header("캐릭터 설정")

    body_action = st.selectbox("몸 (동작)", BODY_ACTIONS)
    hair_kw     = st.selectbox("헤어",     keywords("hair"))
    torso_kw    = st.selectbox("상의",     keywords("torso"))
    legs_kw     = st.selectbox("하의",     keywords("legs"))
    feet_kw     = st.selectbox("발",       keywords("feet"))

    st.divider()
    generate_btn = st.button("Generate", use_container_width=True)

# 초기 실행 또는 Generate 클릭 시 생성
if generate_btn or "sprites" not in st.session_state:
    body_idx  = BODY_ACTIONS.index(body_action)
    hair_idx  = label_map[f"hair_{hair_kw}"]
    torso_idx = label_map[f"torso_{torso_kw}"]
    legs_idx  = label_map[f"legs_{legs_kw}"]
    feet_idx  = label_map[f"feet_{feet_kw}"]

    sprites = {
        "body":  generate_layer(body_model,  body_idx,  body_cfg["num_classes"],  body_cfg["latent_dim"]),
        "hair":  generate_layer(layer_model, hair_idx,  layer_cfg["num_classes"], layer_cfg["latent_dim"]),
        "torso": generate_layer(layer_model, torso_idx, layer_cfg["num_classes"], layer_cfg["latent_dim"]),
        "legs":  generate_layer(layer_model, legs_idx,  layer_cfg["num_classes"], layer_cfg["latent_dim"]),
        "feet":  generate_layer(layer_model, feet_idx,  layer_cfg["num_classes"], layer_cfg["latent_dim"]),
    }
    st.session_state["sprites"] = sprites

sprites = st.session_state["sprites"]

# 레이어 순서대로 합성
result = composite([sprites[cat] for cat in LAYER_ORDER])

# 결과 표시
col_result, col_layers = st.columns([1, 2])

with col_result:
    st.subheader("Result")
    st.image(result.resize((256, 256), Image.NEAREST))

with col_layers:
    st.subheader("Layers")
    cols = st.columns(len(LAYER_ORDER))
    for i, cat in enumerate(LAYER_ORDER):
        with cols[i]:
            st.caption(CATEGORY_KR[cat])
            st.image(arr_to_pil(sprites[cat], 96))

import base64
import io
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

KO_LABELS: dict[str, str] = {
    # 동작
    "walk": "걷기", "idle": "대기", "run": "달리기", "slash": "베기",
    "shoot": "사격", "thrust": "찌르기", "jump": "점프", "sit": "앉기",
    "spellcast": "마법 시전", "other": "기타", "hurt": "피격", "combat": "전투",
    # 헤어
    "afro": "아프로", "balding": "대머리", "bangs": "앞머리컷",
    "bangslong": "긴 앞머리", "bedhead": "헝클어진", "bob": "단발",
    "bun": "번머리", "cornrows": "콘로우", "curly": "곱슬",
    "curtains": "커튼컷", "dreadlocks": "드레드락", "flipped": "플립",
    "flower": "꽃 장식", "french": "프렌치", "hawk": "호크",
    "jewelry": "장신구", "long": "긴머리", "loose": "느슨한",
    "messy": "헝클어진", "mohawk": "모히칸", "page": "페이지컷",
    "parted": "가르마", "pixie": "픽시컷", "plain": "민머리",
    "ponytail": "포니테일", "princess": "공주머리", "shaggy": "샤기",
    "short": "짧은머리", "shoulder": "어깨머리", "side": "사이드컷",
    "spiky": "뾰족머리", "straight": "생머리", "swoop": "스웹",
    "unkempt": "흐트러진", "wavy": "웨이브", "wind": "바람머리",
    # 상의 / 하의 / 발 — 공통 소재·종류
    "plate": "판금", "leather": "가죽", "robe": "로브", "chain": "체인메일",
    "shirt": "셔츠", "vest": "조끼", "cloth": "천", "scale": "비늘갑옷",
    "overalls": "멜빵바지", "pants": "바지", "skirt": "스커트",
    "shorts": "반바지", "boots": "부츠", "shoes": "신발",
    "sandals": "샌들", "bare": "맨발", "slippers": "슬리퍼",
}

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


def pil_to_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_to_content(img: Image.Image, pad: int = 2) -> Image.Image:
    """투명 여백을 제거하고 정사각으로 패딩 (표시용 — 캐릭터를 프레임에 꽉 채움)."""
    bbox = img.getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(img.width, r + pad); b = min(img.height, b + pad)
    cropped = img.crop((l, t, r, b))
    side = max(cropped.width, cropped.height)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    return square


# ── App ──────────────────────────────────────────────────────

st.set_page_config(page_title="캐릭터 생성", layout="wide",
                   initial_sidebar_state="expanded")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Noto+Serif+KR:wght@400;500;700&display=swap');

:root{
  --gold:#d9b25a; --gold-bright:#f2d98a; --gold-dim:#8a6d34;
  --line:rgba(217,178,90,0.40); --line-strong:rgba(217,178,90,0.75);
  --parch:#e8dcc4; --muted:#b8a888;
}

/* 배경: 크림슨 → 블랙 + 비네팅 */
.stApp{
  background:
    radial-gradient(125% 95% at 50% -5%, #4a1010 0%, #240a0b 42%, #0c0607 100%),
    #0c0607;
  color:var(--parch);
  font-family:'Noto Serif KR', serif;
}
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:radial-gradient(85% 75% at 50% 42%, transparent 52%, rgba(0,0,0,0.72) 100%);
}
.block-container{ position:relative; z-index:1; padding-top:4.5rem; }

/* 제목/소제목 */
h1,h2,h3,h4{
  font-family:'Cinzel','Noto Serif KR',serif !important;
  color:var(--gold-bright) !important;
  letter-spacing:0.10em; text-shadow:0 0 18px rgba(217,178,90,0.30);
}

/* 사이드바 = 좌측 설정 패널 */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(34,12,12,0.96), rgba(12,6,7,0.96));
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] *{ color:var(--parch); }

/* 셀렉트박스 */
div[data-baseweb="select"]>div{
  background:rgba(10,6,6,0.85)!important;
  border:1px solid var(--line)!important;
  border-radius:2px!important; color:var(--parch)!important;
}
div[data-baseweb="select"]:hover>div{ border-color:var(--line-strong)!important; }
ul[role="listbox"]{ background:#160a0a!important; border:1px solid var(--line)!important; }

/* 버튼 = 황금 각인 */
.stButton>button, [data-testid="stDownloadButton"]>button{
  background:linear-gradient(180deg,#d6ac52,#7d5e26);
  color:#1a0d04!important;
  font-family:'Cinzel',serif; font-weight:700; letter-spacing:0.22em;
  text-transform:uppercase;
  border:1px solid var(--gold-bright); border-radius:2px;
  box-shadow:0 0 18px rgba(217,178,90,0.40), inset 0 1px 0 rgba(255,255,255,0.35);
  transition:all .15s ease;
}
.stButton>button:hover, [data-testid="stDownloadButton"]>button:hover{
  background:linear-gradient(180deg,#f0cf80,#9a7430);
  box-shadow:0 0 26px rgba(242,217,138,0.6), inset 0 1px 0 rgba(255,255,255,0.45);
}

/* 이미지 = 어두운 비네팅 액자 */
[data-testid="stImage"] img{
  border:1px solid var(--line);
  box-shadow:0 0 28px rgba(0,0,0,0.65);
  background:radial-gradient(circle at 50% 42%, rgba(96,22,22,0.45), rgba(0,0,0,0.9));
}

/* 중앙 캐릭터 무대 — 반응형 + 가운데 정렬 */
.char-stage{ display:flex; justify-content:center; align-items:center; width:100%; }
.char-img{
  width:clamp(190px, 22vw, 360px);
  aspect-ratio:1 / 1;
  image-rendering:pixelated;
  display:block; margin:0 auto;
  border:1px solid var(--line);
  box-shadow:0 0 32px rgba(0,0,0,0.7);
  background:radial-gradient(circle at 50% 42%, rgba(96,22,22,0.45), rgba(0,0,0,0.92));
}

/* 번호 섹션 라벨 */
.sec{ display:flex; align-items:center; gap:0.55rem; margin:0.9rem 0 0.3rem; }
.sec-n{
  display:inline-flex; align-items:center; justify-content:center;
  width:23px; height:23px; flex:none;
  border:1px solid var(--gold); color:var(--gold-bright);
  font-family:'Cinzel',serif; font-size:0.82rem; font-weight:700;
  background:rgba(217,178,90,0.08);
}
.sec-t{ font-family:'Cinzel','Noto Serif KR',serif; color:var(--gold-bright);
        letter-spacing:0.08em; font-size:0.96rem; }

/* 황금 구분선 */
.gold-rule{ height:1px; margin:0.7rem 0 1.2rem;
  background:linear-gradient(90deg,transparent,var(--line-strong),transparent); }

/* 메인 헤더 */
.cc-title{ text-align:center; font-family:'Cinzel','Noto Serif KR',serif;
  font-size:2.1rem; font-weight:700; color:var(--gold-bright);
  letter-spacing:0.34em; text-shadow:0 0 22px rgba(217,178,90,0.45); }
.cc-sub{ text-align:center; color:var(--muted); letter-spacing:0.28em;
  font-size:0.74rem; font-family:'Cinzel',serif; margin-top:0.2rem; }

/* 설명 패널 (parchment) */
.lore{
  border:1px solid var(--line); border-radius:3px;
  background:linear-gradient(180deg, rgba(30,16,12,0.85), rgba(12,7,7,0.85));
  padding:1.3rem 1.4rem; box-shadow:inset 0 0 30px rgba(0,0,0,0.5);
}
.lore h4{ margin:0 0 0.5rem; font-size:1.25rem; }
.lore .row{ display:flex; justify-content:space-between; gap:1rem;
  padding:0.4rem 0; border-bottom:1px solid rgba(217,178,90,0.15);
  font-size:0.9rem; }
.lore .row:last-child{ border-bottom:none; }
.lore .row .k{ color:var(--muted); letter-spacing:0.06em; }
.lore .row .v{ color:var(--parch); font-weight:500; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

body_model, body_cfg   = load_body_model()
layer_model, layer_cfg = load_layer_model()

label_map   = layer_cfg["label_map"]
label_names = layer_cfg["label_names"]


def ko(keyword: str) -> str:
    return KO_LABELS.get(keyword, keyword.replace("_", " ").title())


def keywords(cat: str) -> list[str]:
    prefix = cat + "_"
    return sorted(n[len(prefix):] for n in label_names if n.startswith(prefix))


def section_label(num: int, text: str) -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-n">{num}</span>'
        f'<span class="sec-t">{text}</span></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="sec-t" style="font-size:1.2rem;">캐릭터 설정</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    section_label(1, "몸 · 동작")
    body_action = st.selectbox("body", BODY_ACTIONS, format_func=ko,
                               label_visibility="collapsed")
    section_label(2, "헤어")
    hair_kw     = st.selectbox("hair", keywords("hair"), format_func=ko,
                               label_visibility="collapsed")
    section_label(3, "상의")
    torso_kw    = st.selectbox("torso", keywords("torso"), format_func=ko,
                               label_visibility="collapsed")
    section_label(4, "하의")
    legs_kw     = st.selectbox("legs", keywords("legs"), format_func=ko,
                               label_visibility="collapsed")
    section_label(5, "발")
    feet_kw     = st.selectbox("feet", keywords("feet"), format_func=ko,
                               label_visibility="collapsed")

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
    generate_btn = st.button("⚔  생성", use_container_width=True)

st.markdown('<div class="cc-title">캐릭터 생성</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-sub">CHARACTER CREATION</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

if generate_btn:
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
    st.session_state["choice"]  = {
        "몸 · 동작": ko(body_action), "헤어": ko(hair_kw), "상의": ko(torso_kw),
        "하의": ko(legs_kw), "발": ko(feet_kw),
    }

if "sprites" not in st.session_state:
    st.info("좌측 패널에서 옵션을 선택하고 '생성'을 눌러 캐릭터를 소환하세요.")
    st.stop()

sprites = st.session_state["sprites"]

# 레이어 순서대로 합성
result = composite([sprites[cat] for cat in LAYER_ORDER])

# 결과 표시: 중앙 캐릭터 / 우측 설명 패널
col_char, col_lore = st.columns([1.4, 1])

with col_char:
    st.markdown(
        f'<div class="char-stage"><img class="char-img" '
        f'src="{pil_to_uri(crop_to_content(result))}" alt="character" /></div>',
        unsafe_allow_html=True,
    )

    dl_buf = io.BytesIO()
    result.resize((512, 512), Image.NEAREST).save(dl_buf, format="PNG")
    dc = st.columns([1, 2, 1])
    with dc[1]:
        st.download_button(
            "⬇  이미지 다운로드",
            data=dl_buf.getvalue(),
            file_name="character.png",
            mime="image/png",
            use_container_width=True,
        )

with col_lore:
    choice = st.session_state.get("choice", {})
    rows = "".join(
        f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in choice.items()
    )
    st.markdown(
        f'<div class="lore"><h4>소환된 캐릭터</h4>'
        f'<div style="color:var(--muted);font-size:0.86rem;margin-bottom:0.8rem;">'
        f'CVAE + Perceptual Loss로 생성된 5개 레이어를 알파 합성한 결과입니다.</div>'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-t" style="text-align:center;">레이어 구성</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(LAYER_ORDER))
    for i, cat in enumerate(LAYER_ORDER):
        with cols[i]:
            st.image(arr_to_pil(sprites[cat], 96), use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;color:var(--muted);'
                f'font-size:0.8rem;letter-spacing:0.05em;">{CATEGORY_KR[cat]}</div>',
                unsafe_allow_html=True,
            )

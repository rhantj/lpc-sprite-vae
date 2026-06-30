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
ASSETS     = Path("app_assets/experiments")

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
    "afro": "아프로", "balding": "대머리", "bangs": "앞머리",
    "bangslong": "긴 앞머리", "bangslong2": "긴 앞머리2", "bangsshort": "짧은 앞머리",
    "bedhead": "헝클어진", "bob": "단발", "braid": "땋은머리", "braid2": "땋은머리2",
    "bunches": "양갈래", "buzzcut": "버즈컷", "cornrows": "콘로우",
    "cowlick": "삐죽머리", "curls": "컬", "curly": "곱슬", "curtains": "커튼컷",
    "dreadlocks": "드레드락", "extensions": "익스텐션", "flat": "납작머리",
    "half": "반묶음", "halfmessy": "반묶음(헝클어진)", "high": "높은묶음",
    "idol": "아이돌컷", "jewfro": "유대인 아프로", "lob": "롱단발",
    "long": "긴머리", "longhawk": "긴 호크", "loose": "풀어헤친",
    "messy": "헝클어진", "messy1": "헝클어진1", "messy2": "헝클어진2",
    "messy3": "헝클어진3", "mop": "몹헤어", "natural": "자연스러운",
    "page": "페이지컷", "page2": "페이지컷2", "parted": "가르마",
    "parted2": "가르마2", "parted3": "가르마3", "pigtails": "양갈래",
    "pixie": "픽시컷", "plain": "민머리", "ponytail": "포니테일",
    "ponytail2": "포니테일2", "princess": "공주머리", "relm": "렐름",
    "sara": "사라", "shorthawk": "짧은 호크", "shoulderl": "어깨머리(좌)",
    "shoulderr": "어깨머리(우)", "single": "싱글", "spiked": "뾰족머리",
    "spiked2": "뾰족머리2", "swoop": "스웹", "twists": "트위스트",
    "unkempt": "흐트러진", "wavy": "웨이브", "xlong": "매우 긴머리",
    # 상의
    "aprons": "앞치마", "armour": "갑옷", "bandage": "붕대",
    "chainmail": "체인메일", "clothes": "일반의상", "jacket": "재킷",
    "waist": "허리장식",
    # 하의
    "cuffed": "커프드팬츠", "formal": "정장바지", "fur": "모피",
    "hose": "타이즈", "leggings": "레깅스", "leggings2": "레깅스2",
    "pantaloons": "판탈롱", "pants": "바지", "pants2": "바지2",
    "shorts": "반바지", "skirts": "스커트",
    # 발
    "accessory": "발장식", "boots": "부츠", "hoofs": "발굽",
    "sandals": "샌들", "shoes": "신발", "slippers": "슬리퍼", "socks": "양말",
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


def crop_to_content(img: Image.Image, pad: int = 1,
                    alpha_thresh: int = 40) -> Image.Image:
    """실제 캐릭터(알파>임계값)만 타이트하게 크롭 후 정사각 패딩.
    생성물의 흐릿한 노이즈 픽셀을 무시해 캐릭터가 프레임을 꽉 채우도록 한다."""
    alpha = np.asarray(img)[..., 3]
    ys, xs = np.where(alpha > alpha_thresh)
    if len(xs) == 0:                      # 임계값 통과 픽셀 없으면 원본 bbox로 폴백
        bbox = img.getbbox()
        if not bbox:
            return img
        l, t, r, b = bbox
    else:
        l, t, r, b = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
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
  width:clamp(480px, 56vw, 920px);
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

/* 탭 */
.stTabs [data-baseweb="tab-list"]{ gap:1.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{
  font-family:'Cinzel','Noto Serif KR',serif; color:var(--muted)!important;
  letter-spacing:0.10em; font-size:1.0rem; background:transparent;
}
.stTabs [aria-selected="true"]{ color:var(--gold-bright)!important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--gold)!important; }
.stTabs [data-baseweb="tab-border"]{ background:transparent; }

/* 실험 figure 캡션 */
.exp-cap{ color:var(--muted); font-size:0.84rem; letter-spacing:0.03em;
  margin:0.2rem 0 1.4rem; line-height:1.5; }
.exp-cap b{ color:var(--gold-bright); }

/* 실험 결과 표 */
.exp-table{ width:100%; border-collapse:collapse; margin:0.2rem 0 0.6rem; font-size:0.9rem; }
.exp-table th{ color:var(--gold-bright); text-align:left; font-family:'Cinzel','Noto Serif KR',serif;
  font-weight:600; font-size:0.8rem; letter-spacing:0.05em; padding:0.5rem 0.8rem;
  border-bottom:1px solid var(--line); }
.exp-table td{ color:var(--parch); padding:0.5rem 0.8rem;
  border-bottom:1px solid rgba(217,178,90,0.15); }
.exp-table tr.best td{ background:rgba(217,178,90,0.08); color:var(--gold-bright); font-weight:600; }
.exp-note{ color:var(--muted); font-size:0.82rem; margin:0 0 1.4rem; font-style:italic; }
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


def layer_row(num: int, label: str, key: str, options: list[str],
              lock_enabled: bool):
    """셀렉트박스 + 잠금 체크박스 한 줄. (선택 키워드, 잠금 여부) 반환."""
    section_label(num, label)
    sel_col, lock_col = st.columns([4, 1])
    with sel_col:
        choice = st.selectbox(key, options, format_func=ko,
                              label_visibility="collapsed")
    with lock_col:
        help_txt = ("잠그면 생성 시 이 레이어는 현재 이미지를 유지합니다."
                    if lock_enabled else "먼저 캐릭터를 한 번 생성하면 잠금을 쓸 수 있습니다.")
        locked = st.checkbox("🔒", key=f"lock_{key}",
                             disabled=not lock_enabled, help=help_txt)
    return choice, locked and lock_enabled


with st.sidebar:
    st.markdown('<div class="sec-t" style="font-size:1.2rem;">캐릭터 설정</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # 잠금은 최초 1회 생성 이후부터 활성화
    lock_enabled = "sprites" in st.session_state

    body_action, lock_body  = layer_row(1, "몸 · 동작", "body",  BODY_ACTIONS, lock_enabled)
    hair_kw,     lock_hair  = layer_row(2, "헤어",      "hair",  keywords("hair"), lock_enabled)
    torso_kw,    lock_torso = layer_row(3, "상의",      "torso", keywords("torso"), lock_enabled)
    legs_kw,     lock_legs  = layer_row(4, "하의",      "legs",  keywords("legs"), lock_enabled)
    feet_kw,     lock_feet  = layer_row(5, "발",        "feet",  keywords("feet"), lock_enabled)

    if not lock_enabled:
        st.caption("🔒 잠금은 첫 생성 이후 사용할 수 있습니다.")

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
    generate_btn = st.button("⚔  생성", use_container_width=True)

# 생성 버튼 처리 (활성 탭과 무관하게 매 rerun 실행)
CHOICE_LABEL = {"body": "몸 · 동작", "hair": "헤어", "torso": "상의",
                "legs": "하의", "feet": "발"}

if generate_btn:
    idx = {
        "body":  BODY_ACTIONS.index(body_action),
        "hair":  label_map[f"hair_{hair_kw}"],
        "torso": label_map[f"torso_{torso_kw}"],
        "legs":  label_map[f"legs_{legs_kw}"],
        "feet":  label_map[f"feet_{feet_kw}"],
    }
    kw = {"body": body_action, "hair": hair_kw, "torso": torso_kw,
          "legs": legs_kw, "feet": feet_kw}
    locks = {"body": lock_body, "hair": lock_hair, "torso": lock_torso,
             "legs": lock_legs, "feet": lock_feet}

    def gen_one(cat: str) -> np.ndarray:
        if cat == "body":
            return generate_layer(body_model, idx[cat],
                                  body_cfg["num_classes"], body_cfg["latent_dim"])
        return generate_layer(layer_model, idx[cat],
                              layer_cfg["num_classes"], layer_cfg["latent_dim"])

    prev_sprites = st.session_state.get("sprites", {})
    prev_choice  = st.session_state.get("choice", {})

    sprites, choice, locked_labels = {}, {}, []
    for cat in LAYER_ORDER:
        clabel = CHOICE_LABEL[cat]
        # A 방식: 잠긴 레이어는 이전 이미지를 유지(키워드 변경 무시)
        if locks[cat] and cat in prev_sprites:
            sprites[cat] = prev_sprites[cat]
            choice[clabel] = prev_choice.get(clabel, ko(kw[cat]))
            locked_labels.append(clabel)
        else:
            sprites[cat] = gen_one(cat)
            choice[clabel] = ko(kw[cat])

    st.session_state["sprites"] = sprites
    st.session_state["choice"]  = choice
    st.session_state["locked"]  = locked_labels


# ── Tab renderers ─────────────────────────────────────────────

def tab_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="cc-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cc-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)


def render_generation() -> None:
    tab_header("캐릭터 생성", "CHARACTER CREATION")
    if "sprites" not in st.session_state:
        st.info("좌측 패널에서 옵션을 선택하고 '생성'을 눌러 캐릭터를 소환하세요.")
        return

    sprites = st.session_state["sprites"]
    result = composite([sprites[cat] for cat in LAYER_ORDER])

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
        locked = set(st.session_state.get("locked", []))
        rows = "".join(
            f'<div class="row"><span class="k">{k}</span>'
            f'<span class="v">{v}{" 🔒" if k in locked else ""}</span></div>'
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


def _kv_panel(title: str, items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in items
    )
    return f'<div class="lore"><h4>{title}</h4>{rows}</div>'


def render_architecture() -> None:
    tab_header("모델 구조", "MODEL ARCHITECTURE")
    st.markdown(
        '<div class="lore" style="margin-bottom:1.2rem;"><h4>조건부 변분 오토인코더 (CVAE)</h4>'
        '<div style="color:var(--parch);font-size:0.92rem;line-height:1.65;">'
        '입력 스프라이트와 <b style="color:var(--gold-bright);">레이블(one-hot)</b>을 함께 인코딩해 '
        '잠재변수 z의 분포를 추정하고, z와 레이블을 다시 디코딩해 64×64 RGBA 스프라이트를 복원합니다. '
        '생성 시에는 z ~ N(0,I)를 샘플링하고 원하는 레이블을 주입해 <b style="color:var(--gold-bright);">'
        '원본 없이 순수 생성</b>합니다.</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_kv_panel("Encoder", [
            ("입력", "64×64×4 (RGBA) + label"),
            ("Conv ×4", "stride 2 · 4→32→64→128→256"),
            ("정규화", "BatchNorm + LeakyReLU(0.2)"),
            ("출력", "flatten ⊕ label → μ, logσ² (128-d)"),
        ]), unsafe_allow_html=True)
    with c2:
        st.markdown(_kv_panel("Decoder", [
            ("입력", "z(128) ⊕ label"),
            ("FC", "→ 256×4×4 reshape"),
            ("ConvT ×4", "stride 2 · 256→128→64→32→4"),
            ("출력", "64×64×4, Sigmoid"),
        ]), unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(_kv_panel("Body 모델 (CVAE + Perceptual Loss)", [
            ("latent_dim", "128"),
            ("num_classes", "10 (동작)"),
            ("β (KL 가중치)", "1.5"),
            ("λ_perc (VGG)", "1.0"),
            ("epochs / batch", "50 / 128"),
            ("learning rate", "1e-4"),
        ]), unsafe_allow_html=True)
    with c4:
        st.markdown(_kv_panel("Layer 모델 (단일 CVAE)", [
            ("latent_dim", "128"),
            ("num_classes", "86 (torso·legs·feet·hair)"),
            ("β (KL 가중치)", "1.5"),
            ("max_per_class", "2000"),
            ("epochs / batch", "100 / 128"),
            ("learning rate", "1e-3"),
        ]), unsafe_allow_html=True)

    st.markdown(
        '<div class="lore" style="margin-top:1.2rem;"><h4>손실 함수</h4>'
        '<div class="row"><span class="k">L (전체)</span><span class="v">Recon + β·KL ( + λ·Perceptual )</span></div>'
        '<div class="row"><span class="k">Recon</span><span class="v">입력과 복원 이미지의 픽셀 차이</span></div>'
        '<div class="row"><span class="k">KL</span><span class="v">잠재분포를 prior N(0,I)에 맞추는 정규화</span></div>'
        '<div class="row"><span class="k">Perceptual</span><span class="v">VGG16 feature 차이 (body 모델 한정)</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def exp_figure(file: str, caption_html: str) -> None:
    path = ASSETS / file
    if not path.exists():
        st.warning(f"이미지 없음: {file}")
        return
    st.image(str(path), use_container_width=True)
    st.markdown(f'<div class="exp-cap">{caption_html}</div>', unsafe_allow_html=True)


def metric_table(headers: list[str], rows: list[list[str]],
                 best: int | None = None, note: str = "") -> None:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for i, r in enumerate(rows):
        cls = ' class="best"' if best == i else ""
        trs += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    html = (f'<table class="exp-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{trs}</tbody></table>')
    if note:
        html += f'<div class="exp-note">{note}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_experiments() -> None:
    tab_header("실험 결과", "EXPERIMENT RESULTS")

    section_label(1, "데이터셋 · 증강")
    c = st.columns(2)
    with c[0]:
        exp_figure("dataset.png", "LPC 스프라이트 <b>카테고리 분포</b> — 26개 카테고리, 클래스 불균형 확인")
    with c[1]:
        exp_figure("augment.png", "body <b>좌우반전 증강</b> — 상단 원본 / 하단 반전 (2배 확장)")

    section_label(2, "VAE 재구성 · β-VAE 실험")
    c = st.columns(2)
    with c[0]:
        exp_figure("vae_recon.png", "VAE <b>재구성 결과</b> (PyTorch) — 상단 원본 / 하단 복원")
    with c[1]:
        exp_figure("beta_vae.png", "<b>β별 학습 곡선</b> — β=0.5~4.0 비교, body는 β=1.0이 최적 복원")

    section_label(3, "학습 Loss 분석")
    c = st.columns(2)
    with c[0]:
        exp_figure("vae_loss.png", "VAE <b>학습 곡선</b> — Total / Recon(MSE) / KL Divergence (50 epoch)")
    with c[1]:
        exp_figure("cvae_loss.png", "CVAE <b>학습 곡선</b> — Total Loss, Recon vs KL 균형 (100 epoch)")
    metric_table(
        ["모델 (최종 epoch)", "Total (train)", "Total (val)"],
        [["VAE (PyTorch)", "64.70", "70.48"],
         ["CVAE (PyTorch, β=1.5)", "66.30", "71.44"]],
        note="CVAE 분해(train): Recon 38.44 + β·KL(1.5×18.57) ≈ 66.30 (β=1.5 확인).")

    section_label(4, "잠재공간 시각화")
    c = st.columns(2)
    with c[0]:
        exp_figure("latent_vae.png", "VAE <b>잠재공간</b> (PCA/t-SNE) — 동작별 군집")
    with c[1]:
        exp_figure("latent_cvae.png", "CVAE <b>잠재공간</b> — 레이블 조건부로 더 구조화된 분포")

    section_label(5, "Latent Arithmetic · Interpolation")
    c = st.columns(2)
    with c[0]:
        exp_figure("latent_arithmetic.png", "<b>벡터 산술 (A−B+C)</b> — 잠재 벡터 연산으로 속성 조합")
    with c[1]:
        exp_figure("latent_interp.png", "<b>잠재공간 보간</b> — 방향 벡터를 따라 α=−2→+2로 스윕하며 속성이 연속 변화")

    section_label(6, "CVAE 조건부 생성")
    exp_figure("cvae_gen.png", "CVAE <b>조건부 생성</b> — 레이블별 샘플 (z ~ N(0,I))")

    section_label(7, "레이어 CVAE (86 클래스)")
    c = st.columns(3)
    with c[0]:
        exp_figure("layer_training.png", "<b>학습 곡선</b> — 100 epoch")
    with c[1]:
        exp_figure("layer_recon.png", "<b>재구성</b> 결과")
    with c[2]:
        exp_figure("layer_cond.png", "<b>조건부 생성</b> 결과")
    metric_table(
        ["지표 (최종 epoch)", "값"],
        [["Total Loss · train", "37.30"],
         ["Total Loss · val", "39.35"],
         ["Reconstruction (train)", "21.40"],
         ["KL Divergence (train)", "10.60"]],
        best=0,
        note="86 클래스(torso 7 + legs 12 + feet 8 + hair 59) 단일 CVAE · β=1.5 · 100 epoch. "
             "L = Recon + β·KL = 21.40 + 1.5×10.60 ≈ 37.30.")


# ── Tabs ──────────────────────────────────────────────────────

tab_gen, tab_arch, tab_exp = st.tabs(["⚔  캐릭터 생성", "🏛  모델 구조", "📜  실험 결과"])

with tab_gen:
    render_generation()
with tab_arch:
    render_architecture()
with tab_exp:
    render_experiments()

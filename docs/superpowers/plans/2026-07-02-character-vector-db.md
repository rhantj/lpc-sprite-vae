# 캐릭터 벡터 DB + 유사 검색 (2단계) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 생성한 캐릭터(키워드·설명·latent z·이미지)를 SQLite에 저장하고, 자연어 임베딩 검색으로 과거 캐릭터를 찾아 동일 재현(z 재사용) 또는 변형 생성(z+노이즈)한다.

**Architecture:** 신규 모듈 `character_db.py`가 저장·검색을 담당한다(임베딩 호출은 주입식). Streamlit 앱은 생성 시 레이어별 z를 session_state에 보관하고, 저장 버튼으로 DB에 기록하며, 신규 "갤러리" 탭에서 임베딩 코사인 검색·불러오기·변형 생성을 제공한다. CVAE 모델은 변경하지 않는다.

**Tech Stack:** Python 3.10+, SQLite(표준 라이브러리), Ollama `bge-m3` 임베딩(로컬, 한국어 지원), numpy, pytest, Streamlit

**선행 조건:** 1단계 계획(`2026-07-02-llm-natural-language-generation.md`)의 Task 6까지 완료 상태 — selectbox `sel_{cat}` key, `ollama` 패키지 설치됨.

## Global Constraints

- 임베딩은 로컬 Ollama `bge-m3` 사용. 사전 조건: `ollama pull bge-m3` (약 1.2GB).
- API 키·외부 호출 없음. 모델명은 `character_db.EMBED_MODEL` 상수 한 곳에서만 지정.
- DB 파일은 `characters.db` (프로젝트 루트). git 추적 안 함 → `.gitignore`에 추가.
- Ollama 미가동 시에도 저장·목록·불러오기·변형 생성은 동작해야 함 (임베딩 검색만 비활성, 최신순 폴백).
- 유사도 검색은 numpy 브루트포스 코사인 (수천 건 규모까지 충분 — 벡터 DB 라이브러리 도입 금지, YAGNI).
- 카테고리는 1단계와 동일 5개: `body`, `hair`, `torso`, `legs`, `feet`.
- 테스트에서 실제 Ollama 호출 금지 — 임베딩은 수제 벡터 주입으로 테스트.
- 커밋 메시지는 저장소 관례: `feat:`/`fix:`/`docs:` + 한국어 설명.

---

### Task 1: `character_db.py` — 스키마·저장·목록

**Files:**
- Create: `character_db.py`
- Test: `tests/test_character_db.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `Character` (frozen dataclass): `id: int, description: str, keywords: dict, z_vectors: dict, embedding: list | None, image: bytes, created_at: str`
  - `connect(path: str = DB_PATH) -> sqlite3.Connection` — 스키마 자동 생성
  - `save_character(conn, description: str, keywords: dict, z_vectors: dict, image: bytes, embedding: list | None = None) -> int` — 새 id 반환
  - `list_characters(conn) -> list[Character]` — 최신순

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_character_db.py` 생성:

```python
from character_db import Character, connect, list_characters, save_character

KW = {"body": "run", "hair": "bob", "torso": "armour",
      "legs": "pants", "feet": "boots"}
ZS = {c: [0.1] * 4 for c in KW}          # 테스트용 축소 z
PNG = b"\x89PNG fake bytes"


def make_conn(tmp_path):
    return connect(str(tmp_path / "test.db"))


def test_save_and_list_roundtrip(tmp_path):
    conn = make_conn(tmp_path)
    cid = save_character(conn, "금발 갑옷 전사", KW, ZS, PNG,
                         embedding=[1.0, 0.0])
    chars = list_characters(conn)
    assert cid == 1
    assert len(chars) == 1
    ch = chars[0]
    assert isinstance(ch, Character)
    assert ch.description == "금발 갑옷 전사"
    assert ch.keywords == KW
    assert ch.z_vectors == ZS
    assert ch.embedding == [1.0, 0.0]
    assert ch.image == PNG


def test_save_without_embedding(tmp_path):
    conn = make_conn(tmp_path)
    save_character(conn, "설명", KW, ZS, PNG)
    assert list_characters(conn)[0].embedding is None


def test_list_returns_newest_first(tmp_path):
    conn = make_conn(tmp_path)
    save_character(conn, "첫번째", KW, ZS, PNG)
    save_character(conn, "두번째", KW, ZS, PNG)
    assert [c.description for c in list_characters(conn)] == ["두번째", "첫번째"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/gomuseo/Desktop/Python/vae_test && python -m pytest tests/test_character_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'character_db'`

- [ ] **Step 3: 최소 구현**

`character_db.py` 생성:

```python
"""캐릭터 갤러리 저장소 (SQLite) + 임베딩 유사 검색.

생성 캐릭터의 키워드·설명·레이어별 latent z·합성 이미지를 저장한다.
임베딩 호출은 주입식(embed: Callable[[str], list[float]])이라 provider 교체가 쉽다.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

DB_PATH = "characters.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    keywords TEXT NOT NULL,
    z_vectors TEXT NOT NULL,
    embedding TEXT,
    image BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class Character:
    id: int
    description: str
    keywords: dict
    z_vectors: dict
    embedding: list | None
    image: bytes
    created_at: str


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)  # Streamlit 멀티스레드 대응
    conn.execute(_SCHEMA)
    return conn


def save_character(conn: sqlite3.Connection, description: str,
                   keywords: dict, z_vectors: dict, image: bytes,
                   embedding: list | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO characters (description, keywords, z_vectors, embedding, image) "
        "VALUES (?, ?, ?, ?, ?)",
        (description,
         json.dumps(keywords, ensure_ascii=False),
         json.dumps(z_vectors),
         json.dumps(embedding) if embedding is not None else None,
         image))
    conn.commit()
    return cur.lastrowid


def _to_character(row: tuple) -> Character:
    cid, desc, kw, zs, emb, img, created = row
    return Character(
        id=cid, description=desc,
        keywords=json.loads(kw), z_vectors=json.loads(zs),
        embedding=json.loads(emb) if emb is not None else None,
        image=img, created_at=created)


def list_characters(conn: sqlite3.Connection) -> list[Character]:
    rows = conn.execute(
        "SELECT id, description, keywords, z_vectors, embedding, image, created_at "
        "FROM characters ORDER BY id DESC").fetchall()
    return [_to_character(r) for r in rows]
```

`.gitignore`에 추가:

```bash
grep -qx 'characters.db' .gitignore || echo 'characters.db' >> .gitignore
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_character_db.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add character_db.py tests/test_character_db.py .gitignore
git commit -m "feat: character_db — 캐릭터 SQLite 저장·목록 (키워드·z·이미지)"
```

---

### Task 2: `character_db.py` — 코사인 유사도 검색

**Files:**
- Modify: `character_db.py`
- Test: `tests/test_character_db.py`

**Interfaces:**
- Consumes: `connect`, `save_character`, `Character` (Task 1)
- Produces:
  - `cosine(a: list[float], b: list[float]) -> float`
  - `search(conn, query_embedding: list[float], top_k: int = 6) -> list[tuple[Character, float]]` — 유사도 내림차순, embedding 없는 레코드는 제외

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_character_db.py`에 추가:

```python
import math

from character_db import cosine, search


def test_cosine_basic():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
    assert math.isclose(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)


def test_search_orders_by_similarity(tmp_path):
    conn = make_conn(tmp_path)
    save_character(conn, "정반대", KW, ZS, PNG, embedding=[-1.0, 0.0])
    save_character(conn, "직교", KW, ZS, PNG, embedding=[0.0, 1.0])
    save_character(conn, "일치", KW, ZS, PNG, embedding=[1.0, 0.0])
    results = search(conn, [1.0, 0.0], top_k=2)
    assert [c.description for c, _ in results] == ["일치", "직교"]
    assert results[0][1] > results[1][1]


def test_search_skips_records_without_embedding(tmp_path):
    conn = make_conn(tmp_path)
    save_character(conn, "임베딩 없음", KW, ZS, PNG)
    save_character(conn, "임베딩 있음", KW, ZS, PNG, embedding=[1.0, 0.0])
    results = search(conn, [1.0, 0.0])
    assert [c.description for c, _ in results] == ["임베딩 있음"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_character_db.py -v`
Expected: 신규 3건 FAIL — `ImportError: cannot import name 'cosine'`

- [ ] **Step 3: 구현**

`character_db.py` 상단 import에 `import numpy as np` 추가 후 함수 추가:

```python
def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


def search(conn: sqlite3.Connection, query_embedding: list[float],
           top_k: int = 6) -> list[tuple[Character, float]]:
    scored = [(c, cosine(query_embedding, c.embedding))
              for c in list_characters(conn) if c.embedding is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_character_db.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add character_db.py tests/test_character_db.py
git commit -m "feat: 캐릭터 임베딩 코사인 유사도 검색 (브루트포스 top-k)"
```

---

### Task 3: `character_db.py` — Ollama 임베딩 연결

**Files:**
- Modify: `character_db.py`
- Test: `tests/test_character_db.py`

**Interfaces:**
- Produces: `create_embed(model: str = EMBED_MODEL) -> Callable[[str], list[float]] | None`
  - Ollama 서버 미가동 시 None 반환 (앱은 이를 보고 검색·임베딩 저장 비활성)
  - `EMBED_MODEL = "bge-m3"` 모듈 상수 (교체 지점)

- [ ] **Step 1: 실패하는 테스트 추가** (서버 미가동 경우만 — 실제 호출 금지)

`tests/test_character_db.py`에 추가:

```python
from character_db import create_embed


def test_create_embed_returns_none_when_server_unavailable(monkeypatch):
    import ollama

    def boom():
        raise ConnectionError("server down")

    monkeypatch.setattr(ollama, "list", boom)
    assert create_embed() is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_character_db.py -v`
Expected: 신규 1건 FAIL — `ImportError: cannot import name 'create_embed'`

- [ ] **Step 3: 구현**

`character_db.py` 상단 import에 `from typing import Callable` 추가 후 함수 추가:

```python
EMBED_MODEL = "bge-m3"


def create_embed(model: str = EMBED_MODEL) -> Callable[[str], list[float]] | None:
    import ollama
    try:
        ollama.list()  # 서버 연결 확인
    except Exception:
        return None

    def embed(text: str) -> list[float]:
        resp = ollama.embed(model=model, input=text)
        return list(resp["embeddings"][0])

    return embed
```

- [ ] **Step 4: 테스트 + 스모크 확인**

Run: `python -m pytest tests/test_character_db.py -v`
Expected: 7 passed

Run (모델 준비, 최초 1회 — 약 1.2GB):

```bash
ollama pull bge-m3
```

Run (실제 임베딩 스모크):

```bash
python -c "
from character_db import create_embed, cosine
embed = create_embed()
assert embed is not None, 'Ollama 서버를 먼저 실행하세요 (ollama serve)'
a = embed('갑옷 입은 전사')
b = embed('철갑 기사')
c = embed('꽃무늬 원피스')
print('전사·기사:', round(cosine(a, b), 3), '/ 전사·원피스:', round(cosine(a, c), 3))
assert cosine(a, b) > cosine(a, c)
"
```

Expected: 전사·기사 유사도가 전사·원피스보다 높게 출력.

- [ ] **Step 5: Commit**

```bash
git add character_db.py tests/test_character_db.py
git commit -m "feat: Ollama bge-m3 임베딩 연결 (create_embed)"
```

---

### Task 4: Streamlit — 레이어별 z 보관 및 z 시드 재사용

**Files:**
- Modify: `streamlit_sprite.py` (`generate_layer`, 생성 버튼 처리 블록)

**Interfaces:**
- Consumes: 기존 `generate_layer`, 생성 블록(`if generate_btn:`)
- Produces:
  - `generate_layer(..., z: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor]` — z 미지정 시 랜덤 샘플, 사용한 z 반환
  - `st.session_state["zs"]: dict[str, list[float]]` — 카테고리별 사용된 z
  - `st.session_state["kw"]: dict[str, str]` — 카테고리별 영문 키워드 (DB 저장용)
  - 생성 트리거: `generate_btn or st.session_state.pop("auto_generate", False)`
  - z 시드 소비: `st.session_state.pop("z_seeds", {})` (`dict[str, list[float]]`, 갤러리 탭이 채움)

- [ ] **Step 1: `generate_layer`에 z 파라미터 추가**

기존 (`streamlit_sprite.py:154`):

```python
def generate_layer(model, label_idx: int, num_classes: int, latent_dim: int) -> np.ndarray:
    z = torch.randn(1, latent_dim).to(DEVICE)
    c = F.one_hot(torch.tensor([label_idx]), num_classes).float().to(DEVICE)
    with torch.no_grad():
        img = model.decoder(z, c)
    return img[0].cpu().permute(1, 2, 0).numpy()
```

변경:

```python
def generate_layer(model, label_idx: int, num_classes: int, latent_dim: int,
                   z: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor]:
    if z is None:
        z = torch.randn(1, latent_dim).to(DEVICE)
    c = F.one_hot(torch.tensor([label_idx]), num_classes).float().to(DEVICE)
    with torch.no_grad():
        img = model.decoder(z, c)
    return img[0].cpu().permute(1, 2, 0).numpy(), z
```

- [ ] **Step 2: 생성 블록에서 z 수집·시드 소비**

`if generate_btn:` 블록을 다음으로 교체 (auto_generate 트리거, z_seeds 소비, zs/kw 저장 추가):

```python
if generate_btn or st.session_state.pop("auto_generate", False):
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
    z_seeds = st.session_state.pop("z_seeds", {})

    def gen_one(cat: str) -> tuple[np.ndarray, torch.Tensor]:
        seed = z_seeds.get(cat)
        z = (torch.tensor([seed], dtype=torch.float32).to(DEVICE)
             if seed is not None else None)
        if cat == "body":
            return generate_layer(body_model, idx[cat],
                                  body_cfg["num_classes"], body_cfg["latent_dim"], z=z)
        return generate_layer(layer_model, idx[cat],
                              layer_cfg["num_classes"], layer_cfg["latent_dim"], z=z)

    prev_sprites = st.session_state.get("sprites", {})
    prev_choice  = st.session_state.get("choice", {})
    prev_zs      = st.session_state.get("zs", {})
    prev_kw      = st.session_state.get("kw", {})

    sprites, choice, zs, kws, locked_labels = {}, {}, {}, {}, []
    for cat in LAYER_ORDER:
        clabel = CHOICE_LABEL[cat]
        # A 방식: 잠긴 레이어는 이전 이미지를 유지(키워드 변경 무시)
        if locks[cat] and cat in prev_sprites:
            sprites[cat] = prev_sprites[cat]
            choice[clabel] = prev_choice.get(clabel, ko(kw[cat]))
            zs[cat] = prev_zs.get(cat)
            kws[cat] = prev_kw.get(cat, kw[cat])
            locked_labels.append(clabel)
        else:
            sprites[cat], z_used = gen_one(cat)
            choice[clabel] = ko(kw[cat])
            zs[cat] = z_used[0].cpu().tolist()
            kws[cat] = kw[cat]

    st.session_state["sprites"] = sprites
    st.session_state["choice"]  = choice
    st.session_state["locked"]  = locked_labels
    st.session_state["zs"]      = zs
    st.session_state["kw"]      = kws
```

- [ ] **Step 3: 수동 검증**

Run: `streamlit run streamlit_sprite.py`
확인:
1. 기존과 동일하게 생성 동작 (회귀 없음: 생성·잠금·다운로드)
2. 생성 후 파이썬 콘솔 없이 확인이 어려우므로 임시로 `st.caption(str(len(st.session_state['zs'])))`를 넣어 5가 표시되는지 본 뒤 제거해도 됨

Run: `python -m pytest tests/ -v`
Expected: 전체 PASS (기존 테스트 회귀 없음)

- [ ] **Step 4: Commit**

```bash
git add streamlit_sprite.py
git commit -m "feat: 생성 시 레이어별 latent z 보관 및 z 시드 재사용 지원"
```

---

### Task 5: Streamlit — 갤러리 저장 UI

**Files:**
- Modify: `streamlit_sprite.py` (모듈 상단 연결, `render_generation`)

**Interfaces:**
- Consumes: `connect`, `save_character`, `create_embed` (Task 1–3), `st.session_state["zs"|"kw"|"sprites"]` (Task 4)
- Produces: `DB`(전역 커넥션), `EMBED`(전역, None 가능), 생성 탭 하단 저장 UI

- [ ] **Step 1: 모듈 상단 연결 코드 추가**

import 블록에 추가:

```python
from character_db import (connect, create_embed, list_characters,
                          save_character, search)
```

`LLM_CALL = get_llm_call()` 아래에 추가:

```python
@st.cache_resource
def get_db():
    return connect()


@st.cache_resource
def get_embed():
    return create_embed()


DB = get_db()
EMBED = get_embed()
```

- [ ] **Step 2: `render_generation`에 저장 UI 추가**

`render_generation` 함수 끝부분(다운로드 버튼 아래)에 추가:

```python
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
    save_desc = st.text_input(
        "캐릭터 설명 (검색에 사용됩니다)", key="save_desc",
        placeholder="예: 금발 단발머리 갑옷 전사")
    if st.button("💾 갤러리에 저장", use_container_width=True):
        if not save_desc.strip():
            st.warning("설명을 입력해 주세요.")
        else:
            base = composite([sprites[c] for c in LAYER_ORDER])
            buf = io.BytesIO()
            base.save(buf, format="PNG")
            emb = EMBED(save_desc) if EMBED is not None else None
            save_character(DB, save_desc, st.session_state["kw"],
                           st.session_state["zs"], buf.getvalue(), emb)
            msg = "저장 완료" if emb is not None else \
                  "저장 완료 (Ollama 미가동 — 이 캐릭터는 자연어 검색에서 제외됩니다)"
            st.success(msg)
```

주의: `sprites`는 `render_generation` 안에서 이미 `st.session_state["sprites"]`로 조회돼 있음 — 해당 지역변수 사용.

- [ ] **Step 3: 수동 검증**

Run: `streamlit run streamlit_sprite.py`
확인:
1. 생성 → 설명 입력 → 저장 → "저장 완료" 표시
2. `sqlite3 characters.db "SELECT id, description FROM characters;"` 로 행 확인
3. Ollama 중지 후 저장 → 임베딩 제외 안내 문구 확인

- [ ] **Step 4: Commit**

```bash
git add streamlit_sprite.py
git commit -m "feat: 생성 캐릭터를 갤러리 DB에 저장 (설명 임베딩 포함)"
```

---

### Task 6: Streamlit — 갤러리 탭 (검색·불러오기·변형 생성)

**Files:**
- Modify: `streamlit_sprite.py` (탭 정의, 신규 `render_gallery`)

**Interfaces:**
- Consumes: `DB`, `EMBED`, `list_characters`, `search` (Task 5), `z_seeds`/`auto_generate` 소비 로직 (Task 4)
- Produces: 4번째 탭 "🖼 갤러리". 불러오기 = 저장된 z 그대로 재현, 변형 생성 = z + 0.3·N(0,1)

- [ ] **Step 1: `render_gallery` 함수 추가**

`render_experiments` 함수 아래에 추가:

```python
VARIATION_NOISE = 0.3  # 변형 생성 시 z에 더할 가우시안 노이즈 배율


def _apply_character(ch, noise: float = 0.0) -> None:
    """저장된 캐릭터를 드롭다운·z 시드에 적용하고 자동 생성을 예약한다."""
    for cat, kw in ch.keywords.items():
        st.session_state[f"sel_{cat}"] = kw
    seeds = {}
    for cat, z in ch.z_vectors.items():
        if z is None:
            continue
        arr = np.asarray(z, dtype=np.float32)
        if noise:
            arr = arr + noise * np.random.randn(*arr.shape).astype(np.float32)
        seeds[cat] = arr.tolist()
    st.session_state["z_seeds"] = seeds
    st.session_state["auto_generate"] = True
    st.rerun()


def render_gallery() -> None:
    tab_header("갤러리", "CHARACTER GALLERY")
    query = st.text_input(
        "자연어로 검색", key="gallery_query",
        placeholder="예: 갑옷 입은 전사", disabled=EMBED is None)
    if EMBED is None:
        st.caption("Ollama 미가동 — 검색이 비활성화되어 최신순으로 표시합니다.")

    if query.strip() and EMBED is not None:
        results = search(DB, EMBED(query), top_k=6)
        chars = [c for c, _ in results]
        scores = {c.id: s for c, s in results}
    else:
        chars = list_characters(DB)
        scores = {}

    if not chars:
        st.info("저장된 캐릭터가 없습니다. 캐릭터 생성 탭에서 저장해 보세요.")
        return

    cols = st.columns(3)
    for i, ch in enumerate(chars):
        with cols[i % 3]:
            st.image(ch.image, width=160)
            score = f" · 유사도 {scores[ch.id]:.2f}" if ch.id in scores else ""
            st.caption(f"#{ch.id} {ch.description}{score}")
            c1, c2 = st.columns(2)
            if c1.button("불러오기", key=f"load_{ch.id}",
                         use_container_width=True):
                _apply_character(ch)
            if c2.button("변형 생성", key=f"var_{ch.id}",
                         use_container_width=True):
                _apply_character(ch, noise=VARIATION_NOISE)
```

- [ ] **Step 2: 탭 등록**

기존:

```python
tab_gen, tab_arch, tab_exp = st.tabs(["⚔  캐릭터 생성", "🏛  모델 구조", "📜  실험 결과"])
```

변경 (기존 `with tab_...:` 블록들 아래에 갤러리 블록 추가):

```python
tab_gen, tab_gallery, tab_arch, tab_exp = st.tabs(
    ["⚔  캐릭터 생성", "🖼  갤러리", "🏛  모델 구조", "📜  실험 결과"])
```

그리고 기존 탭 렌더 블록들과 같은 위치에:

```python
with tab_gallery:
    render_gallery()
```

- [ ] **Step 3: 수동 검증**

Run: `streamlit run streamlit_sprite.py`
확인:
1. 캐릭터 2~3개 저장 후 갤러리 탭에 그리드 표시
2. "갑옷 전사" 검색 → 관련 캐릭터가 상위, 유사도 표시
3. **불러오기** → 생성 탭에서 저장 당시와 동일한 캐릭터 재현 (z 재사용 확인)
4. **변형 생성** → 비슷하지만 조금 다른 캐릭터 생성 (노이즈 확인)
5. Ollama 중지 → 검색 비활성 안내 + 최신순 목록, 불러오기/변형은 정상 동작

Run: `python -m pytest tests/ -v`
Expected: 전체 PASS

- [ ] **Step 4: Commit**

```bash
git add streamlit_sprite.py
git commit -m "feat: 갤러리 탭 — 자연어 검색·불러오기·z 노이즈 변형 생성"
```

---

### Task 7: 문서화

**Files:**
- Modify: `README.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1: README 업데이트**

"자연어 입력 (로컬 LLM)" 섹션 아래에 추가:

```markdown
### 캐릭터 갤러리 (벡터 DB)

생성한 캐릭터를 저장하고 자연어로 다시 찾을 수 있다.

- 저장: 키워드 + 설명 + 레이어별 latent z + 합성 이미지 → `characters.db` (SQLite, git 미추적)
- 검색: 설명을 Ollama `bge-m3` 임베딩으로 변환해 코사인 유사도 top-k (`ollama pull bge-m3` 필요)
- 불러오기: 저장된 z를 그대로 디코딩해 동일 캐릭터 재현
- 변형 생성: z + 0.3·N(0,1) 노이즈로 비슷하지만 다른 캐릭터 생성

Ollama가 없으면 검색만 비활성화되고 저장·불러오기·변형은 동작한다.
```

- [ ] **Step 2: PROGRESS.md 업데이트**

"LLM 연계" 섹션 아래에 추가:

```markdown
### 캐릭터 벡터 DB (2단계)
- [x] `character_db.py` — SQLite 저장 + bge-m3 임베딩 코사인 검색
- [x] 갤러리 탭 — 자연어 검색 / z 재사용 재현 / z 노이즈 변형 생성
```

- [ ] **Step 3: Commit**

```bash
git add README.md PROGRESS.md
git commit -m "docs: 캐릭터 갤러리(벡터 DB) 사용법 및 진행 상황 기록"
```

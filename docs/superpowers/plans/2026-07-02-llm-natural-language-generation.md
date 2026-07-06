# LLM 자연어 캐릭터 생성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 자연어 문장 한 줄을 로컬 LLM(Ollama + qwen3:8b)으로 5개 레이어 키워드(JSON)로 변환해 기존 CVAE 생성 파이프라인에 연결한다.

**Architecture:** 신규 모듈 `llm_mapper.py`가 자연어 → 키워드 매핑을 담당한다(순수 함수 + 주입 가능한 LLM 호출 함수). Streamlit 앱은 텍스트 입력을 받아 매핑 결과로 드롭다운(session_state)을 세팅하고 기존 생성 로직을 그대로 재사용한다. CVAE 파이프라인은 변경하지 않는다.

**Tech Stack:** Python 3.10+, Ollama (qwen3:8b, 로컬), ollama Python SDK, pytest, Streamlit (기존 앱)

## Global Constraints

- LLM은 로컬 Ollama 서버의 `qwen3:8b` 모델 사용. 사전 조건: Ollama 설치 + `ollama pull qwen3:8b`.
- API 키·외부 API 호출 없음. 모델명은 `llm_mapper.OLLAMA_MODEL` 상수 한 곳에서만 지정 (교체 지점).
- 허용 키워드는 하드코딩하지 않고 `checkpoints_cvae_layer_pt/config.json`의 `label_names`에서 동적 로드 (body 동작 10종은 `streamlit_sprite.py`의 `BODY_ACTIONS` 사용).
- LLM 출력에서 허용 목록 밖 키워드가 최종 결과로 나오면 안 됨 (검증 레이어에서 null 처리).
- Ollama 서버 미가동/호출 실패 시에도 기존 드롭다운 워크플로우는 정상 동작해야 함.
- 카테고리는 5개 고정: `body`, `hair`, `torso`, `legs`, `feet`.
- 커밋 메시지는 저장소 관례를 따름: `feat:`/`fix:`/`docs:` + 한국어 설명.
- 테스트에서 실제 LLM 호출 금지 — fake call 함수 주입으로 테스트.

---

### Task 1: `llm_mapper.py` — 허용 키워드 로드 + 검증

**Files:**
- Create: `llm_mapper.py`
- Test: `tests/test_llm_mapper.py`

**Interfaces:**
- Produces:
  - `CATEGORIES: tuple[str, ...] = ("body", "hair", "torso", "legs", "feet")`
  - `load_allowed(label_names: list[str], body_actions: list[str]) -> dict[str, list[str]]`
  - `validate(data: dict, allowed: dict[str, list[str]]) -> dict[str, str | None]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_llm_mapper.py` 생성:

```python
from llm_mapper import CATEGORIES, load_allowed, validate

LABEL_NAMES = [
    "hair_bangs", "hair_long", "torso_armour", "torso_clothes",
    "legs_pants", "legs_skirts", "feet_boots", "feet_shoes",
]
BODY_ACTIONS = ["walk", "idle", "run"]
ALLOWED = load_allowed(LABEL_NAMES, BODY_ACTIONS)


def test_load_allowed_builds_category_dict():
    assert ALLOWED == {
        "body": ["walk", "idle", "run"],
        "hair": ["bangs", "long"],
        "torso": ["armour", "clothes"],
        "legs": ["pants", "skirts"],
        "feet": ["boots", "shoes"],
    }


def test_validate_keeps_valid_keywords():
    data = {"body": "run", "hair": "bangs", "torso": "armour",
            "legs": "pants", "feet": "boots"}
    assert validate(data, ALLOWED) == data


def test_validate_nulls_invalid_keywords():
    data = {"body": "fly", "hair": "bangs", "torso": None,
            "legs": 123, "feet": "boots"}
    result = validate(data, ALLOWED)
    assert result == {"body": None, "hair": "bangs", "torso": None,
                      "legs": None, "feet": "boots"}


def test_validate_handles_missing_keys_and_non_dict():
    assert validate({}, ALLOWED) == {c: None for c in CATEGORIES}
    assert validate("garbage", ALLOWED) == {c: None for c in CATEGORIES}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /Users/gomuseo/Desktop/Python/vae_test && python -m pytest tests/test_llm_mapper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_mapper'`

- [ ] **Step 3: 최소 구현**

`llm_mapper.py` 생성:

```python
"""자연어 → CVAE 키워드 매핑 (LLM 어댑터).

허용 키워드는 체크포인트 config의 label_names에서 동적 로드한다.
LLM 호출 함수는 주입식(call: Callable[[str], str])이라 provider 교체가 쉽다.
"""
from __future__ import annotations

CATEGORIES: tuple[str, ...] = ("body", "hair", "torso", "legs", "feet")


def load_allowed(label_names: list[str],
                 body_actions: list[str]) -> dict[str, list[str]]:
    allowed: dict[str, list[str]] = {"body": list(body_actions)}
    for cat in CATEGORIES[1:]:
        prefix = cat + "_"
        allowed[cat] = sorted(
            n[len(prefix):] for n in label_names if n.startswith(prefix))
    return allowed


def validate(data: object,
             allowed: dict[str, list[str]]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for cat in CATEGORIES:
        kw = data.get(cat) if isinstance(data, dict) else None
        out[cat] = kw if isinstance(kw, str) and kw in allowed[cat] else None
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_mapper.py tests/test_llm_mapper.py
git commit -m "feat: llm_mapper 모듈 — 허용 키워드 로드 및 검증"
```

---

### Task 2: `llm_mapper.py` — 프롬프트 구성

**Files:**
- Modify: `llm_mapper.py`
- Test: `tests/test_llm_mapper.py`

**Interfaces:**
- Consumes: `CATEGORIES`, `load_allowed` (Task 1)
- Produces: `build_prompt(text: str, allowed: dict[str, list[str]], ko_labels: dict[str, str]) -> str`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_llm_mapper.py`에 추가:

```python
from llm_mapper import build_prompt

KO = {"run": "달리기", "bangs": "앞머리", "armour": "갑옷", "boots": "부츠"}


def test_build_prompt_contains_user_text_and_keywords():
    prompt = build_prompt("갑옷 입고 달리는 캐릭터", ALLOWED, KO)
    assert "갑옷 입고 달리는 캐릭터" in prompt
    assert "run(달리기)" in prompt          # 한글 라벨 병기
    assert "armour(갑옷)" in prompt
    assert "long" in prompt                  # 라벨 없는 키워드는 그대로
    for cat in CATEGORIES:
        assert cat in prompt


def test_build_prompt_mentions_json_and_null():
    prompt = build_prompt("아무거나", ALLOWED, KO)
    assert "JSON" in prompt
    assert "null" in prompt
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 신규 2건 FAIL — `ImportError: cannot import name 'build_prompt'`

- [ ] **Step 3: 구현**

`llm_mapper.py`에 추가:

```python
def build_prompt(text: str, allowed: dict[str, list[str]],
                 ko_labels: dict[str, str]) -> str:
    lines = []
    for cat in CATEGORIES:
        kws = ", ".join(
            f"{k}({ko_labels[k]})" if k in ko_labels else k
            for k in allowed[cat])
        lines.append(f"- {cat}: {kws}")
    keyword_block = "\n".join(lines)
    return f"""당신은 픽셀 캐릭터 생성기의 키워드 매핑기입니다.
사용자의 캐릭터 설명을 읽고, 카테고리별로 아래 허용 키워드 중 가장 적합한 것을 하나씩 고르세요.

허용 키워드 (키워드(한글 설명) 형식):
{keyword_block}

규칙:
1. 반드시 허용 키워드 목록에 있는 영문 키워드만 사용하세요.
2. 설명에 언급되지 않은 카테고리는 null로 두세요.
3. 아래 형식의 JSON 객체만 출력하세요. 다른 텍스트 금지.

{{"body": "...", "hair": "...", "torso": "...", "legs": "...", "feet": "..."}}

사용자 설명: {text}"""
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add llm_mapper.py tests/test_llm_mapper.py
git commit -m "feat: llm_mapper 프롬프트 구성 (허용 키워드 + 한글 라벨 주입)"
```

---

### Task 3: `llm_mapper.py` — `map_text` (호출 + 파싱 + 재시도)

**Files:**
- Modify: `llm_mapper.py`
- Test: `tests/test_llm_mapper.py`

**Interfaces:**
- Consumes: `build_prompt`, `validate` (Task 1, 2)
- Produces: `map_text(text: str, allowed: dict[str, list[str]], ko_labels: dict[str, str], call: Callable[[str], str]) -> dict[str, str | None]`
  - `call`은 프롬프트 문자열을 받아 LLM 응답 문자열을 반환하는 함수 (주입식)
  - 반환값: 카테고리별 검증된 키워드 또는 None. 완전 실패 시 전체 None.

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_llm_mapper.py`에 추가:

```python
import json

from llm_mapper import map_text

GOOD_JSON = json.dumps({"body": "run", "hair": "bangs", "torso": "armour",
                        "legs": None, "feet": None})


def test_map_text_happy_path():
    result = map_text("갑옷 전사", ALLOWED, KO, call=lambda p: GOOD_JSON)
    assert result == {"body": "run", "hair": "bangs", "torso": "armour",
                      "legs": None, "feet": None}


def test_map_text_retries_on_invalid_json_then_succeeds():
    responses = iter(["not json {", GOOD_JSON])
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return next(responses)

    result = map_text("갑옷 전사", ALLOWED, KO, call=fake_call)
    assert result["body"] == "run"
    assert len(calls) == 2


def test_map_text_retries_on_invalid_keyword_with_feedback():
    bad = json.dumps({"body": "fly", "hair": "bangs", "torso": None,
                      "legs": None, "feet": None})
    responses = iter([bad, GOOD_JSON])
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return next(responses)

    result = map_text("나는 전사", ALLOWED, KO, call=fake_call)
    assert result["body"] == "run"
    assert "fly" in calls[1]                 # 재시도 프롬프트에 오류 피드백 포함


def test_map_text_returns_all_none_after_two_failures():
    result = map_text("전사", ALLOWED, KO, call=lambda p: "garbage")
    assert result == {c: None for c in CATEGORIES}


def test_map_text_second_attempt_invalid_keyword_becomes_none():
    bad = json.dumps({"body": "fly", "hair": "bangs", "torso": None,
                      "legs": None, "feet": None})
    result = map_text("전사", ALLOWED, KO, call=lambda p: bad)
    assert result == {"body": None, "hair": "bangs", "torso": None,
                      "legs": None, "feet": None}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 신규 5건 FAIL — `ImportError: cannot import name 'map_text'`

- [ ] **Step 3: 구현**

`llm_mapper.py` 상단 import에 추가:

```python
import json
from typing import Callable
```

함수 추가:

```python
def map_text(text: str, allowed: dict[str, list[str]],
             ko_labels: dict[str, str],
             call: Callable[[str], str]) -> dict[str, str | None]:
    prompt = build_prompt(text, allowed, ko_labels)
    for attempt in range(2):
        raw = call(prompt)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        result = validate(data, allowed)
        invalid = {c: data.get(c) for c in CATEGORIES
                   if isinstance(data, dict)
                   and data.get(c) is not None and result[c] is None}
        if invalid and attempt == 0:
            prompt = build_prompt(text, allowed, ko_labels) + (
                "\n\n이전 응답에 허용되지 않은 키워드가 있었습니다: "
                f"{json.dumps(invalid, ensure_ascii=False)}. "
                "반드시 허용 목록의 키워드만 사용하세요.")
            continue
        return result
    return {c: None for c in CATEGORIES}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add llm_mapper.py tests/test_llm_mapper.py
git commit -m "feat: map_text — LLM 호출·JSON 파싱·검증·1회 재시도"
```

---

### Task 4: Ollama 클라이언트 연결 + 의존성

**Files:**
- Modify: `llm_mapper.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `create_call(model: str = OLLAMA_MODEL) -> Callable[[str], str] | None`
  - Ollama 서버에 연결 불가하면 None 반환 (앱은 이를 보고 기능 비활성화)
  - `OLLAMA_MODEL = "qwen3:8b"` 모듈 상수 (교체 지점)

- [ ] **Step 1: 실패하는 테스트 추가** (서버 미가동 경우만 단위 테스트 — 실제 LLM 호출 금지)

`tests/test_llm_mapper.py`에 추가:

```python
from llm_mapper import create_call


def test_create_call_returns_none_when_server_unavailable(monkeypatch):
    import ollama

    def boom():
        raise ConnectionError("server down")

    monkeypatch.setattr(ollama, "list", boom)
    assert create_call() is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run (선행): `pip install ollama`
Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 신규 1건 FAIL — `ImportError: cannot import name 'create_call'`

- [ ] **Step 3: 구현**

`llm_mapper.py`에 함수 추가:

```python
OLLAMA_MODEL = "qwen3:8b"


def create_call(model: str = OLLAMA_MODEL) -> Callable[[str], str] | None:
    import ollama
    try:
        ollama.list()  # 서버 연결 확인
    except Exception:
        return None

    def call(prompt: str) -> str:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",   # JSON 출력 강제
            think=False,     # qwen3 thinking 모드 비활성 (응답 속도)
        )
        return resp["message"]["content"]

    return call
```

`requirements.txt` 끝에 추가:

```
ollama
```

- [ ] **Step 4: 테스트 + 스모크 확인**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 12 passed

Run (모델 준비, 최초 1회 — 약 5GB 다운로드):

```bash
ollama pull qwen3:8b
```

Run (실제 매핑 스모크):

```bash
python -c "
import json
from pathlib import Path
from llm_mapper import load_allowed, map_text, create_call
from streamlit_sprite import BODY_ACTIONS, KO_LABELS
cfg = json.loads(Path('checkpoints_cvae_layer_pt/config.json').read_text())
allowed = load_allowed(cfg['label_names'], BODY_ACTIONS)
call = create_call()
assert call is not None, 'Ollama 서버를 먼저 실행하세요 (ollama serve)'
print(map_text('금발 단발머리에 갑옷 입고 달리는 캐릭터', allowed, KO_LABELS, call))
"
```

Expected: `{'body': 'run', 'hair': 'bob', 'torso': 'armour', ...}` 형태의 dict (키워드는 모두 허용 목록 내).

주의: `streamlit_sprite` import 시 torch/streamlit이 로드됨 — 느리지만 동작함. 실패 시 BODY_ACTIONS를 직접 리스트로 넣어 확인해도 됨.

- [ ] **Step 5: Commit**

```bash
git add llm_mapper.py tests/test_llm_mapper.py requirements.txt
git commit -m "feat: Ollama 로컬 LLM 연결 (create_call, qwen3:8b) 및 의존성 추가"
```

---

### Task 5: 노트북 13 — 매핑 실험/정확도 평가

**Files:**
- Create: `13_llm_prompt_mapping.ipynb`

**Interfaces:**
- Consumes: `load_allowed`, `map_text`, `create_call`, `OLLAMA_MODEL` (Task 1–4)

- [ ] **Step 1: 노트북 작성**

`13_llm_prompt_mapping.ipynb`를 아래 셀 구성으로 생성 (마크다운 셀 → 코드 셀 순):

**셀 1 (markdown):**

```markdown
# 13. LLM 프롬프트 매핑 실험

자연어 문장 → 레이어 키워드 매핑(`llm_mapper.map_text`)의 정확도를 평가한다.

- LLM: Ollama 로컬 `qwen3:8b` (사전 조건: `ollama serve` 실행 + `ollama pull qwen3:8b`)
- 평가: 테스트 문장 20개에 대해 카테고리별 키워드 일치율 측정
- 성공 기준: 매핑 정확도 80% 이상, 무효 키워드 0건
```

**셀 2 (code) — 셋업:**

```python
import json
from pathlib import Path

from llm_mapper import (CATEGORIES, OLLAMA_MODEL, create_call,
                        load_allowed, map_text)

BODY_ACTIONS = ["walk", "idle", "run", "slash", "shoot",
                "thrust", "jump", "sit", "spellcast", "other"]

cfg = json.loads(Path("checkpoints_cvae_layer_pt/config.json").read_text())
allowed = load_allowed(cfg["label_names"], BODY_ACTIONS)

# 한글 라벨은 앱과 동일 소스 사용
from streamlit_sprite import KO_LABELS

call = create_call()
assert call is not None, "Ollama 서버를 먼저 실행하세요 (ollama serve)"
print(f"model={OLLAMA_MODEL}, 카테고리별 키워드 수:",
      {c: len(v) for c, v in allowed.items()})
```

**셀 3 (code) — 평가 세트:** 문장과 기대 매핑. 기대값이 여러 개 가능하면 set으로 표기. 언급 없는 카테고리는 None.

```python
# (문장, {카테고리: 정답 키워드 집합 또는 None})
EVAL_SET = [
    ("금발 단발머리에 갑옷 입고 달리는 캐릭터",
     {"body": {"run"}, "hair": {"bob", "lob"}, "torso": {"armour"},
      "legs": None, "feet": None}),
    ("포니테일에 재킷 입고 걷는 사람",
     {"body": {"walk"}, "hair": {"ponytail", "ponytail2"},
      "torso": {"jacket"}, "legs": None, "feet": None}),
    ("치마 입고 샌들 신고 앉아있는 캐릭터",
     {"body": {"sit"}, "hair": None, "torso": None,
      "legs": {"skirts"}, "feet": {"sandals"}}),
    ("긴머리에 체인메일 입고 검을 휘두르는 전사",
     {"body": {"slash"}, "hair": {"long", "xlong"},
      "torso": {"chainmail"}, "legs": None, "feet": None}),
    ("마법을 시전하는 드레드락 머리 캐릭터",
     {"body": {"spellcast"}, "hair": {"dreadlocks"}, "torso": None,
      "legs": None, "feet": None}),
    ("반바지에 부츠 신고 점프하는 캐릭터",
     {"body": {"jump"}, "hair": None, "torso": None,
      "legs": {"shorts"}, "feet": {"boots"}}),
    ("활 쏘는 양갈래 머리 소녀",
     {"body": {"shoot"}, "hair": {"pigtails", "bunches"}, "torso": None,
      "legs": None, "feet": None}),
    ("앞치마 두르고 슬리퍼 신고 서 있는 캐릭터",
     {"body": {"idle"}, "hair": None, "torso": {"aprons"},
      "legs": None, "feet": {"slippers"}}),
    ("정장바지에 구두 신고 걷는 신사",
     {"body": {"walk"}, "hair": None, "torso": None,
      "legs": {"formal"}, "feet": {"shoes"}}),
    ("창으로 찌르는 버즈컷 병사",
     {"body": {"thrust"}, "hair": {"buzzcut"}, "torso": None,
      "legs": None, "feet": None}),
    # ... 같은 형식으로 10개 이상 추가해 총 20개 이상 구성
]
```

**셀 4 (code) — 실행 및 채점:**

```python
rows, n_correct, n_total, n_invalid = [], 0, 0, 0
for text, expected in EVAL_SET:
    result = map_text(text, allowed, KO_LABELS, call)
    for cat in CATEGORIES:
        exp, got = expected[cat], result[cat]
        ok = (got is None) if exp is None else (got in exp)
        n_correct += ok
        n_total += 1
        if got is not None and got not in allowed[cat]:
            n_invalid += 1
        if not ok:
            rows.append((text, cat, exp, got))

print(f"정확도: {n_correct}/{n_total} = {n_correct / n_total:.1%}")
print(f"무효 키워드: {n_invalid}건")
print("\n오답 목록:")
for r in rows:
    print(r)
```

**셀 5 (markdown) — 결과 기록:** 실행 후 정확도·오답 경향·프롬프트 개선 내역을 기록.

- [ ] **Step 2: 노트북 실행 및 프롬프트 반복 개선**

Run: 노트북 전체 실행 (Ollama 서버 가동 상태에서)
Expected: 정확도 80% 이상, 무효 키워드 0건. 미달 시 `build_prompt`의 규칙 문구를 보강하고 (수정 시 Task 2 테스트 재실행: `python -m pytest tests/test_llm_mapper.py -v` → 전체 PASS 유지) 재평가. 프롬프트 개선으로도 미달이면 `OLLAMA_MODEL`을 `exaone3.5:7.8b`로 교체해 재평가.

- [ ] **Step 3: Commit**

```bash
git add 13_llm_prompt_mapping.ipynb
git commit -m "feat: 노트북 13 — LLM 매핑 정확도 평가 (N% 달성)"
```

(N은 실제 측정값으로 교체)

---

### Task 6: Streamlit 통합 — 자연어 입력 UI

**Files:**
- Modify: `streamlit_sprite.py` (import 블록, `layer_row`, 사이드바)

**Interfaces:**
- Consumes: `load_allowed`, `map_text`, `create_call` (Task 1–4)
- Produces: 사이드바 텍스트 입력 → 드롭다운 자동 세팅. selectbox는 `key="sel_{cat}"`로 session_state 제어 가능해짐.

- [ ] **Step 1: llm_mapper 연결 코드 추가**

`streamlit_sprite.py`의 import 블록(파일 상단 `from PIL import Image` 아래)에 추가:

```python
from llm_mapper import create_call, load_allowed, map_text
```

`label_names = layer_cfg["label_names"]` 라인 바로 아래에 추가:

```python
ALLOWED = load_allowed(label_names, BODY_ACTIONS)


@st.cache_resource
def get_llm_call():
    return create_call()


LLM_CALL = get_llm_call()
```

- [ ] **Step 2: selectbox에 명시적 key 부여**

`layer_row` 함수의 selectbox 호출을 수정 (session_state로 값을 세팅하려면 명시적 key 필요):

기존:

```python
        choice = st.selectbox(key, options, format_func=ko,
                              label_visibility="collapsed")
```

변경:

```python
        choice = st.selectbox(key, options, format_func=ko,
                              label_visibility="collapsed",
                              key=f"sel_{key}")
```

- [ ] **Step 3: 사이드바에 자연어 입력 블록 추가**

사이드바에서 `layer_row` 호출들(`body_action, lock_body = layer_row(...)`) **앞**, `lock_enabled = "sprites" in st.session_state` 라인 앞에 추가 (위젯 생성 전에 session_state를 세팅해야 하므로 반드시 layer_row보다 먼저 실행):

```python
    # 자연어 → 키워드 매핑 (LLM)
    desc = st.text_input(
        "문장으로 설명", key="llm_text",
        placeholder="예: 금발 단발머리에 갑옷 입고 달리는 캐릭터",
        disabled=LLM_CALL is None)
    llm_btn = st.button("✨ 문장으로 설정", use_container_width=True,
                        disabled=LLM_CALL is None)
    if LLM_CALL is None:
        st.caption("Ollama 서버에 연결할 수 없어 자연어 입력이 비활성화되었습니다. "
                   "아래 드롭다운은 그대로 사용할 수 있습니다.")
    if llm_btn and desc.strip():
        try:
            mapping = map_text(desc, ALLOWED, KO_LABELS, LLM_CALL)
        except Exception:
            mapping = None
            st.error("LLM 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.")
        if mapping is not None:
            if all(v is None for v in mapping.values()):
                st.warning("문장을 해석하지 못했습니다. 다르게 표현해 보세요.")
            else:
                for cat, kw in mapping.items():
                    if kw is not None:
                        st.session_state[f"sel_{cat}"] = kw
                matched = ", ".join(ko(kw) for kw in mapping.values()
                                    if kw is not None)
                st.caption(f"적용됨: {matched}")
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)
```

- [ ] **Step 4: 단위 테스트 회귀 + 수동 검증**

Run: `python -m pytest tests/test_llm_mapper.py -v`
Expected: 12 passed

Run: `streamlit run streamlit_sprite.py` 후 브라우저에서 확인:
1. Ollama 가동 중: "금발 단발머리에 갑옷 입고 달리는 캐릭터" 입력 → 버튼 클릭 → 드롭다운이 달리기/단발/갑옷 등으로 바뀜 → "생성" 클릭 시 해당 캐릭터 생성
2. 언급 없는 레이어(하의/발)는 기존 드롭다운 값 유지 확인
3. Ollama 서버 중지 후 앱 재실행: 입력·버튼 비활성 + 안내 문구, 드롭다운 생성은 정상 동작
4. 기존 기능 회귀 확인: 레이어 잠금, 생성, 다운로드 정상

- [ ] **Step 5: Commit**

```bash
git add streamlit_sprite.py
git commit -m "feat: 자연어 문장으로 캐릭터 설정 (LLM 매핑 UI)"
```

---

### Task 7: 문서화

**Files:**
- Modify: `README.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1: README 업데이트**

`README.md`의 "Streamlit 앱" 섹션 표 아래에 추가:

```markdown
### 자연어 입력 (로컬 LLM)

문장 한 줄로 캐릭터를 설정할 수 있다. 로컬 Ollama(qwen3:8b)가 문장을 키워드 조합으로 변환한다.

1. [Ollama](https://ollama.com) 설치
2. 모델 다운로드 (최초 1회, 약 5GB):
   ```bash
   ollama pull qwen3:8b
   ```
3. 앱 실행 후 사이드바 상단 입력창에 문장 입력 → "문장으로 설정"

Ollama가 실행 중이 아니면 자연어 입력만 비활성화되고 드롭다운 방식은 그대로 동작한다.
API 키·외부 호출 없이 완전 로컬로 동작하며, 원격 배포 환경에서는 이 기능이 자동 비활성화된다.
```

노트북 표에 행 추가:

```markdown
| 13 | `13_llm_prompt_mapping.ipynb` | LLM 자연어 → 키워드 매핑 정확도 평가 |
```

- [ ] **Step 2: PROGRESS.md 업데이트**

`## 완료` 섹션 끝에 추가 (정확도는 Task 5 실측값으로 교체):

```markdown
### LLM 연계 (자연어 캐릭터 생성)
- [x] `llm_mapper.py` — 자연어 → 키워드 매핑 (Ollama qwen3:8b, 검증·재시도 포함)
- [x] 매핑 정확도 평가 (`13`) — 테스트 문장 세트 N% / 무효 키워드 0건
- [x] Streamlit 자연어 입력 UI (Ollama 미가동 시 드롭다운 폴백)
```

- [ ] **Step 3: Commit**

```bash
git add README.md PROGRESS.md
git commit -m "docs: LLM 자연어 캐릭터 생성 사용법 및 진행 상황 기록"
```

# LLM 연계 확장 계획서 — 자연어 캐릭터 생성

작성일: 2026-07-02

## 1. 개요

기존 LPC Sprite CVAE 프로젝트(키워드 드롭다운 → 레이어별 CVAE 생성 → 합성)에
LLM을 연결해 **자연어 문장 한 줄로 캐릭터를 생성**하는 기능을 추가한다.

```
"금발 단발머리에 갑옷 입고 달리는 캐릭터"
        ↓  LLM (structured output, JSON)
{ body: "run", hair: "bangs", torso: "armour", legs: "pants", feet: "boots" }
        ↓  기존 CVAE 파이프라인 (변경 없음)
     레이어별 생성 → alpha composite → 캐릭터 이미지
```

핵심 원칙: **CVAE 파이프라인은 건드리지 않는다.** LLM은 자연어를
기존 조건(키워드 조합)으로 변환하는 번역기 역할만 한다.

## 2. 목표 / 비목표

### 목표
- 자연어 입력 → 5개 레이어 키워드 매핑 (body 동작 10종, hair/torso/legs/feet 86 클래스)
- Streamlit 앱에 텍스트 입력 UI 통합 (기존 드롭다운과 공존)
- 매핑 정확도 정량 평가 (테스트 문장 세트)

### 비목표 (이번 범위 제외)
- 대화형 점진 수정 ("머리 더 길게") — 3단계 후보 (8절)
- 캐릭터 서사/페르소나 생성 — 이후 확장 후보
- CVAE 모델 재학습

## 3. LLM 선택

| 항목 | 결정 |
|------|------|
| 기본 | **Ollama + qwen3:8b (로컬)** — API 키·비용·rate limit 없음, JSON 출력 강제(`format="json"`) 지원 |
| 구조 | provider 교체 가능한 얇은 어댑터 (Gemini/Claude 등 API로 교체 용이) |
| 사전 조건 | Ollama 설치 + `ollama pull qwen3:8b` (메모리 약 6GB) |
| 배포 제약 | 원격 배포 환경에는 Ollama 서버가 없으므로 자연어 기능은 로컬 전용. 배포 앱은 드롭다운만 동작 |

## 4. 아키텍처

새로 추가되는 것은 모듈 1개 + 노트북 1개 + 앱 UI 1블록.

```
llm_mapper.py              # 신규 모듈 (앱과 노트북에서 공용)
├── ALLOWED_KEYWORDS       # 체크포인트 label_names에서 로드한 카테고리별 허용 키워드
├── build_prompt(text)     # 허용 키워드 목록을 주입한 시스템 프롬프트 구성
├── map_text(text) -> dict # LLM 호출(Ollama) + JSON 파싱 + 검증
└── validate(result)       # 허용 목록 밖 키워드 → 재시도 1회 → 실패 시 None

13_llm_prompt_mapping.ipynb  # 매핑 실험/평가 노트북
streamlit_sprite.py          # 텍스트 입력 → 드롭다운 자동 세팅 → 생성
```

### 매핑 규칙
- LLM 출력은 JSON 스키마로 강제 (레이어별 키워드, 없으면 null)
- 허용 키워드 목록은 하드코딩하지 않고 체크포인트의 `label_names`에서 동적 로드
  (앱의 `keywords()` 함수와 동일 소스 → 모델 갱신 시 자동 반영)
- 문장에 언급 없는 레이어: null → 앱에서 현재 드롭다운 값 유지 (사용자가 이해하기 쉬움)
- 검증 실패 키워드: 재시도 1회, 그래도 실패하면 해당 레이어만 null 처리

### 에러 처리
- Ollama 서버 미가동 / 호출 실패: 텍스트 입력 비활성화 + 안내 문구, 기존 드롭다운 방식은 그대로 동작 (기능 저하일 뿐 앱은 정상)
- 응답 JSON 파싱 실패: 재시도 1회 후 사용자에게 실패 메시지

## 5. 구현 단계

### Phase 1 — 매핑 모듈 + 실험 (노트북 13)
1. `llm_mapper.py` 작성: 허용 키워드 로드, 프롬프트 구성, Ollama 호출, 검증
2. `13_llm_prompt_mapping.ipynb`: 테스트 문장 20~30개(한국어 위주, 영어 일부)로
   매핑 결과 확인, 프롬프트 반복 개선
3. 평가 지표: 레이어별 매핑 정확도(정답 키워드 일치율), 무효 키워드 발생률

### Phase 2 — Streamlit 통합
4. 사이드바 상단에 텍스트 입력 + "문장으로 생성" 버튼 추가
5. 매핑 결과로 5개 selectbox 값 세팅(session_state) 후 기존 생성 로직 재사용
6. 매핑된 키워드를 한글 라벨로 표시해 사용자가 확인/수정 가능하게
7. Ollama 미가동 시 폴백 UI

### Phase 3 — 평가 및 문서화
8. 테스트 문장 세트 정확도 측정 결과를 experiment_log/README에 기록
9. README에 사용법(Ollama 설치·모델 pull 포함) 추가, PROGRESS.md 갱신
10. requirements.txt에 `ollama` 추가

## 6. 성공 기준
- 테스트 문장 세트에서 레이어 매핑 정확도 80% 이상
- 무효 키워드(허용 목록 밖) 최종 출력 0건 (검증 레이어에서 차단)
- Ollama 미가동 시에도 기존 드롭다운 워크플로우 정상 동작

## 7. 리스크

| 리스크 | 대응 |
|--------|------|
| LPC 키워드가 영어 축약형(예: `bangs`, `hose`)이라 LLM이 혼동 | 프롬프트에 키워드별 한글 설명(기존 `KO_LABELS` 재사용) 포함 |
| 8B 로컬 모델의 매핑 품질 | 노트북 13 평가 세트로 검증. 80% 미달 시 exaone3.5:7.8b 또는 Gemini API로 교체 (어댑터만 교체) |
| provider 종속 | 어댑터 함수 1개만 교체하면 되도록 호출부 격리 |
| 86 클래스 전부 프롬프트에 넣으면 길어짐 | 카테고리별로 정리해 주입(수백 토큰 수준, 문제 없음) |

## 8. 확장 로드맵

### 2단계 — 캐릭터 벡터 DB + 유사 검색 (계획 확정)

생성한 캐릭터를 저장하고 자연어로 다시 찾아 재현·변형하는 RAG 구조.
상세 구현 계획: `docs/superpowers/plans/2026-07-02-character-vector-db.md`

```
저장: 키워드 + 설명 + 레이어별 latent z + 합성 PNG → SQLite (characters.db)
검색: 설명 임베딩(Ollama bge-m3) → 코사인 유사도 top-k (numpy 브루트포스)
재현: 저장된 z를 그대로 디코딩 → 동일 캐릭터 복원
변형: z + 0.3·N(0,1) → 비슷하지만 다른 캐릭터
```

- 텍스트 임베딩 검색과 CVAE 잠재공간(z 재사용)이 결합되는 것이 핵심
- Ollama 미가동 시 검색만 비활성 (저장·불러오기·변형은 동작)
- 신규 모듈 `character_db.py` + Streamlit 갤러리 탭. 예상 작업량 7태스크 (약 1~1.5일)

### 3단계 — 대화형 에이전트 (후보, 미확정)

LLM이 tool(`set_layers`, `search_similar`, `regenerate_layer`, 잠금)을 호출하며
멀티턴 대화로 캐릭터를 점진 수정. qwen3:8b tool-calling 품질이 리스크
(필요 시 qwen3:14b). 예상 작업량 5~7태스크 (약 1.5~2일)

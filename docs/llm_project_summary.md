# LLM 연계 프로젝트 요약

작성일: 2026-07-02 · 상태: **계획 수립 완료, 구현 전**

## 한 줄 요약

LPC 스프라이트 CVAE 프로젝트에 로컬 LLM(Ollama)을 연결해,
**① 자연어 문장으로 캐릭터 생성 → ② 캐릭터 벡터 DB 저장·검색·변형**까지 확장한다.

## 전체 그림

```
"금발 단발머리에 갑옷 입고 달리는 캐릭터"
        ↓ qwen3:8b (자연어 → 키워드 JSON)          [1단계]
{body: run, hair: bob, torso: armour, ...}
        ↓ 기존 CVAE 파이프라인 (변경 없음)
     캐릭터 이미지 + 레이어별 latent z
        ↓ 저장 (SQLite: 키워드·설명·z·PNG)          [2단계]
"갑옷 입은 전사 비슷한 애" → bge-m3 임베딩 검색
        → 불러오기(z 재현) / 변형 생성(z + 노이즈)
```

## 단계별 내용

| 단계 | 내용 | 산출물 | 규모 |
|------|------|--------|------|
| 1 | 자연어 → 키워드 매핑 | `llm_mapper.py`, 노트북 13(정확도 평가), 앱 텍스트 입력 UI | 7태스크, ~1일 |
| 2 | 캐릭터 벡터 DB + 유사 검색 | `character_db.py`, 앱 갤러리 탭(저장·검색·재현·변형) | 7태스크, ~1.5일 |
| 3 (후보) | tool-use 대화형 에이전트 | 미확정 — 채팅으로 점진 수정 | 5~7태스크 |

## 핵심 설계 결정

- **LLM**: Ollama + `qwen3:8b` (로컬, API 키·비용 없음). 호출 함수 주입식이라 provider 교체는 함수 1개
- **임베딩**: Ollama + `bge-m3` (한국어 지원). 유사도는 numpy 코사인 브루트포스 (YAGNI)
- **CVAE는 건드리지 않음**: LLM은 번역기, DB는 z 저장소 — 생성 파이프라인 재사용
- **폴백 우선**: Ollama 미가동 시 자연어 입력·검색만 비활성, 기존 드롭다운·저장·재현은 정상 동작
- **검증 레이어**: LLM 출력은 허용 키워드 목록(체크포인트 `label_names` 동적 로드)으로 검증, 무효 키워드 차단

## 성공 기준

- 매핑 정확도 80% 이상 (테스트 문장 20개, 노트북 13에서 측정)
- 무효 키워드 최종 출력 0건
- 갤러리 불러오기 시 저장 당시 캐릭터 동일 재현 (z 재사용)
- Ollama 미가동 시에도 앱 정상 동작

## 사전 준비물

```bash
ollama pull qwen3:8b   # 1단계, 약 5GB
ollama pull bge-m3     # 2단계, 약 1.2GB
pip install ollama
```

## 문서 링크

- 설계 계획서: `docs/llm_extension_plan.md`
- 1단계 구현 계획: `docs/superpowers/plans/2026-07-02-llm-natural-language-generation.md`
- 2단계 구현 계획: `docs/superpowers/plans/2026-07-02-character-vector-db.md`

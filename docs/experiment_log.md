# 실험 로그

## 데이터 전처리 (03, 03-1)

### 카테고리 재분류

원본 body 카테고리에 혼재해 있던 항목들을 분리:

| 분류 작업 | 프레임 수 | 결과 |
|-----------|-----------|------|
| wings 분리 | 100,264 | `processed/wings/` |
| tail 분리 | 93,590 | `processed/tail/` |
| zombie + skeleton 삭제 | 356 | 삭제 |
| wound 분리 | 2,528 | `processed/wound/` |
| wheelchair 분리 | 70 | `processed/wheelchair/` |
| prosthesis 분리 | 532 | `processed/prosthesis/` |

최종 body: 2,272장 (순수 인체 body만)

### 데이터 증강 (좌우반전)

body 카테고리의 좌우반전 이미지를 추가해 데이터셋 2배 확장.

- 파일명 규칙: `{원본 stem}_flip.png`
- 처리 방식: `PIL.Image.FLIP_LEFT_RIGHT` + `ThreadPoolExecutor(max_workers=8)`
  - macOS Jupyter에서 `multiprocessing.Pool` 사용 시 spawn 충돌 발생 → Thread 방식으로 우회
- 결과: 2,272 → **4,544장**

---

## 발생한 문제와 해결

### 1. `buffer_size must be greater than zero` (TF shuffle 에러)

- **원인**: `categories = 'body'` (문자열)로 설정 시 `for cat in 'body'`가 `b`, `o`, `d`, `y` 문자 순회 → 경로 0개 → `shuffle(buffer_size=0)`
- **해결**: `categories = ['body']` (리스트)로 수정. 빈 경로 시 ValueError + CWD 정보 출력 추가

### 2. `TypeError: EagerTensor object is not callable` (LR 콜백)

- **원인**: Keras 3 (TF 2.16)에서 `optimizer.learning_rate`가 호출 가능한 스케줄 객체가 아닌 EagerTensor를 반환
- **해결**: `optimizer.learning_rate(optimizer.iterations)` → `float(lr_schedule(optimizer.iterations))`로 스케줄 객체를 직접 호출

### 3. `KeyError: 'val_loss'` (학습 곡선 플롯)

- **원인**: Keras 3 동작 변경으로 history 키 존재 여부가 불확실
- **해결**: 플롯 셀에 `if f'val_{key}' in hist:` 가드 추가

### 4. `ZeroDivisionError` (PyTorch 검증 루프)

- **원인**: `val_loader`에 `drop_last=True` 설정 + body val 샘플 수(45) < batch_size(64) → 배치 0개
- **해결**: `val_loader`를 `drop_last=False`로 변경, `vn = max(len(val_loader), 1)` 가드 추가

### 5. CFG 저장 시 정수 누락 (`isinstance` 버그)

- **원인**: `isinstance(v, type(tf.data.AUTOTUNE))` → `tf.data.AUTOTUNE`은 정수 -1이므로 `type(...)` = `int` → 모든 정수 CFG 값(epochs, batch_size 등) 제외됨
- **해결**: `_exclude = {'prefetch', 'parallel_calls'}` 명시적 집합으로 교체

### 6. 빈 PNG 파일로 학습 크래시

- **원인**: `dataset/processed/body/`에 0바이트 PNG 3개 존재
- **해결**: 해당 파일 삭제

### 7. 학습 속도 과다 (70분/에폭)

- **원인**: `categories = None`으로 전체 2.8M 프레임 로드 → 21,814 스텝/에폭
- **해결**: `categories = ['body']`로 변경 → 4,544프레임, 35 스텝/에폭

---

## Posterior Collapse 해결

### 현상

어떤 입력을 넣어도 복원 결과가 동일한 "평균 이미지" 출력.

### 진단 수치 (수정 전)

| 지표 | 값 | 의미 |
|------|-----|------|
| kl_loss | ~0.001 | 사실상 0 → 인코더가 입력 무시 |
| μ 절댓값 평균 | ~0.02 | 항상 prior 근처 |
| σ 평균 | ~1.000 | prior N(0,I)와 동일 |

### 원인

`F.mse_loss(reduction='mean')`: B×C×H×W = 16,384개 원소 평균  
`torch.mean(KL)`: B×latent_dim = 256개 원소 평균  
→ Recon이 KL보다 ~64배 크게 정규화되어 KL이 학습에 기여하지 못함

### 해결

```python
# 수정 전
recon_loss = F.mse_loss(recon, x, reduction='mean')
kl_loss    = -0.5 * torch.mean(...)

# 수정 후
batch      = x.size(0)
recon_loss = F.mse_loss(recon, x, reduction='sum') / batch
kl_loss    = -0.5 * torch.sum(...) / batch
```

TF 버전도 동일하게 `tf.reduce_sum(...) / batch` 로 수정.

### 해결 후 수치

| 지표 | 값 |
|------|-----|
| kl_loss | 35~39 (활성화) |
| μ 절댓값 평균 | 0.30~0.37 (다양화) |
| σ 평균 | 0.87~0.96 (인코더 작동) |

---

## TF ↔ PyTorch 결과 차이 원인 분석

두 노트북의 복원 결과가 다른 주요 원인:

| 원인 | 상세 |
|------|------|
| **BatchNorm momentum** | TF 기본 0.99 vs PyTorch 기본 0.1 → running stats 갱신 속도 10배 차이 |
| **가중치 초기화** | TF Glorot uniform vs PyTorch Kaiming uniform (LeakyReLU에는 Kaiming이 이론적 우위) |
| **LR 업데이트 단위** | TF: 스텝마다 / PyTorch: 에폭마다 |
| **Padding 구현** | TF `padding='same'` vs PyTorch `padding=1` (경계 픽셀 처리 방식 차이) |

### 통일 조치

TF 노트북의 BatchNorm momentum을 `0.9`로 수정 (PyTorch `momentum=0.1`과 동일한 동작).

---

## β-VAE 실험 결과

실험 범위: β = 0.5, 1.0, 2.0, 4.0

**최종 결정: β=1.0**

body 데이터셋은 동질성이 높은 인체 형태 위주라 β를 높일수록 KL이 잠재 공간을 과도하게 압축해 디테일이 뭉개짐. β=1.0(표준 VAE)이 시각적으로 가장 좋은 복원 품질을 보임.

---

## Latent Space Arithmetic (06, 06-1)

VAE 잠재 공간에서 벡터 연산으로 이미지 속성을 제어하는 기법.

### 구현 내용

| 섹션 | 기능 |
|------|------|
| 4. 기본 산술 | `z_A - z_B + z_C` 연산 및 시각화 |
| 5. 방향 벡터 추출 | 파일명 키워드로 두 그룹을 나눠 `mean(z_A) - mean(z_B)` 방향 추출 |
| 6. 속성 이식 | `z_대상 + α × 방향벡터` (α = -2 ~ +2, 실행마다 랜덤 이미지) |
| 7. 다중 방향 합성 | 두 방향벡터 동시 적용 (실행마다 랜덤 이미지) |

### 실험 결과

**섹션 6 — 속성 이식**
- α 부호에 따라 원본 이미지가 방향벡터가 가리키는 속성 쪽으로 변화하는 것 확인
- 원본과 전이 목표 속성 간 차이가 클수록(α 절댓값이 클수록) 복원 이미지가 뭉개지는 현상 발생
  - 원인: VAE 잠재 공간의 유효 범위를 벗어나면 디코더가 의미 없는 이미지를 생성함
  - α를 작은 값(0.5~1.0)부터 시작해 점진적으로 늘리는 것이 적절

**섹션 7 — 다중 벡터 합성**
- `idle` 방향벡터를 합성하면 결과물이 더 매끄럽게 출력되는 효과 확인
  - idle 포즈 특성상 동작이 없고 정적인 이미지 → 잠재 공간에서 노이즈가 적은 영역에 위치
  - 해당 방향을 더하면 디코더 출력이 안정적인 영역으로 이동해 이미지가 정돈되는 효과

**idle 방향벡터가 smoothing으로 작용하는 이유**
- `walk`, `run`, `slash` 등 동작 포즈는 프레임마다 팔다리 위치가 달라 분산이 큼 → 그룹 평균 z의 신뢰도 낮음
- `idle`은 프레임 간 차이가 거의 없어 그룹 평균 z가 안정적인 대표점이 됨
- VAE KL 손실로 정적·단순한 이미지일수록 z가 원점 근처에 분포 → idle 방향은 디코더가 잘 학습된 영역에 가까움
- 결과적으로 `dir_idle`이 정규화 역할을 해 다른 방향벡터 적용으로 유효 범위를 벗어나려는 z를 안정 영역으로 당겨주는 효과
  ```
  z_결과 = z_원본 + α × dir_other + α × dir_idle
                                    ↑ 디코더 안정 영역으로 당기는 힘
  ```

### 사용 가능한 키워드 (body 데이터셋)

| 종류 | 키워드 |
|------|--------|
| 캐릭터 타입 | `male`, `female`, `teen`, `child`, `pregnant`, `muscular` |
| 동작 | `walk`, `run`, `jump`, `slash`, `shoot`, `thrust`, `idle`, `sit`, `spellcast`, `climb`, `hurt`, `emote` |

---

## Latent Space 시각화 (07)

PCA / t-SNE / UMAP으로 128차원 잠재 벡터를 2D로 축소해 분포를 시각화.

### 방법

| 방법 | 특성 |
|------|------|
| PCA | 선형 축소, 전체 분산 파악 |
| t-SNE | 비선형, 클러스터 구조 파악 |
| UMAP | 비선형, 전역 구조 보존, t-SNE보다 빠름 |

### 주요 결과

**idle smoothing 가설 수치 확인**

| 그룹 | 평균 std |
|------|---------|
| idle | 0.1292 (가장 낮음) |
| walk | 0.1446 |
| run  | 0.1627 (가장 높음) |

idle이 잠재 공간에서 가장 촘촘하게 모여 있음을 수치로 확인 — 06 실험의 smoothing 효과 원인을 뒷받침.

**전체 분포**: mean≈0 (정상), std=0.36 (prior 1.0보다 낮음 — KL이 prior를 완전히 따라잡지 못한 상태, 일반적인 현상)

**캐릭터 타입 / 액션 클러스터**: PCA·t-SNE·UMAP 모두 색이 섞여 있고 뚜렷한 분리 없음
- 원인: 레이블 없이 비지도학습으로 학습해 의미 있는 속성이 자동 분리된다는 보장 없음

**UMAP 유의사항**: 동일한 Z 벡터를 다른 색으로만 칠한 것이라 점 분포 자체는 두 그래프(캐릭터/액션)가 동일하게 보임 — 색 클러스터 여부만 해석해야 함

---

## CVAE 잠재 공간 시각화 (07-1)

CVAE(PyTorch β=1.5) 잠재 벡터를 07 VAE와 동일한 방법(PCA / t-SNE)으로 시각화해 비교.

### 클러스터 분리

PCA·t-SNE 모두 액션/캐릭터 분리 없음 — VAE(07)와 동일한 결과.

CVAE 인코더는 레이블을 입력받지만, z는 "레이블로 설명 안 되는 나머지 정보"만 담으려 하기 때문에 z 자체엔 액션 구분 정보가 오히려 줄어드는 경향이 있음. 조건부 생성 품질은 디코더의 레이블 주입에 달려 있으며, 잠재 공간의 클러스터 분리 여부와는 별개.

### idle smoothing — VAE vs CVAE 수치 비교

| group | CVAE std | VAE std (07) |
|-------|---------|-------------|
| idle  | **0.1179** | 0.1292 |
| walk  | **0.1263** | 0.1446 |
| run   | **0.1578** | 0.1627 |

- CVAE가 VAE보다 전반적으로 std 낮음 → β=1.5의 강한 KL 정규화로 z가 더 조밀하게 압축됨
- idle의 최저 std 패턴은 CVAE에서도 유지 → smoothing 역할은 β·모델 종류와 무관한 데이터 특성
- 전체 std: CVAE 0.32 < VAE 0.36 → 잠재 공간 구멍 현상이 감소한 원인과 일치

---

## CVAE (Conditional VAE) 실험 (08, 08-1)

VAE 인코더·디코더 모두에 action 레이블(one-hot)을 concat해 조건 주입.

### 설정

| 항목 | 값 |
|------|-----|
| 레이블 | action 9종 + other = 10 클래스 |
| 주입 방식 | conv 출력 flatten 후 one-hot concat |
| epochs | 100 |
| batch_size | 128 |
| latent_dim | 128 |
| LR | cosine annealing (1e-3 → 1e-5) |
| early stopping | patience=10 |

### 학습 결과 (β=1.0, 100 에폭)

| 모델 | Best val_loss | 최종 recon | 최종 KL |
|------|--------------|-----------|---------|
| TF   | 59.47 | 33.05 | 21.79 |
| PyTorch | 61.54 | 33.40 | 23.49 |

50 에폭 대비 val_loss 약 20 개선. 100 에폭 끝에서도 완만히 감소 중 → 추가 학습 여지 있음.

### β 실험 (CVAE)

| β | TF | PyTorch |
|---|-----|---------|
| 1.0 | 복원 일부 뭉개짐, 조건부 생성 불안정 | 복원 양호 |
| 1.5 | 복원·생성 모두 악화 | 복원 유지, 조건부 생성 안정화 (구멍 현상 해소) |

**최종 결정**: TF β=1.0 / PyTorch β=1.5

### 잠재 공간 구멍(hole) 현상

조건부 생성에서 N(0,I) 순수 샘플링 시 일부 z가 학습 분포 밖에 착지 → 뭉개진 이미지 생성.

- PyTorch β=1.5: KL 정규화 강화로 잠재 공간이 더 빽빽하게 채워져 구멍 현상 해소
- TF β=1.5: 오히려 복원 품질 악화 → β=1.0 유지

### 레이블 반응 한계

조건부 생성에서 레이블 간 포즈 차이가 미미함.

- 원인: CVAE가 레이블을 조건으로 쓰면서도 이미지 정보 대부분을 z에 담아버리는 경향
- 개선 방향: 레이블 임베딩 강화, 데이터 확충, 또는 KL 가중치 조정

---

## 현재 구현 현황

| 기법 | 상태 |
|------|------|
| 표준 VAE | 완료 (TF + PyTorch) |
| Latent space interpolation | 완료 (양쪽 섹션 10) |
| β-VAE | 완료 — β=1.0 최종 선택 |
| Latent space arithmetic | 완료 (TF: 06 / PyTorch: 06-1) |
| Latent space 시각화 (VAE) | 완료 (07) — PCA, t-SNE, UMAP |
| Latent space 시각화 (CVAE) | 완료 (07-1) — VAE vs CVAE std 비교 |
| CVAE | 완료 (TF: 08 β=1.0 / PyTorch: 08-1 β=1.5) |
| Perceptual loss | 미구현 |

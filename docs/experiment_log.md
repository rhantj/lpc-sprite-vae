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

## 현재 구현 현황

| 기법 | 상태 |
|------|------|
| 표준 VAE | 완료 (TF + PyTorch) |
| Latent space interpolation | 완료 (양쪽 섹션 10) |
| β-VAE | 실험 노트북(05) 완료, 학습 대기 |
| CVAE | 미구현 |
| Latent space arithmetic | 미구현 |
| Latent space 시각화 (t-SNE/PCA) | 미구현 |
| Perceptual loss | 미구현 |

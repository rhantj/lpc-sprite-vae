# 모델 아키텍처

## VAE 개요

Variational Autoencoder(VAE)는 인코더가 입력을 잠재 분포 N(μ, σ²)으로 압축하고, 디코더가 샘플링된 z에서 이미지를 복원하는 생성 모델입니다.

```
입력 x (64×64×4)
    │
    ▼
[ Encoder ]
  Conv2D × 4 (stride=2)
  64 → 32 → 16 → 8 → 4
    │
    ├─ Dense → μ  (128차원)
    └─ Dense → log σ²  (128차원)
          │
          ▼  재매개변수화: z = μ + ε·σ  (ε ~ N(0,I))
    │
    ▼
[ Decoder ]
  Dense → Reshape (4×4×256)
  ConvTranspose2D × 4 (stride=2)
  4 → 8 → 16 → 32 → 64
    │
    ▼
복원 x̂ (64×64×4)  [Sigmoid 출력 → [0, 1]]
```

## 인코더 레이어 구성

| 레이어 | 입력 크기 | 출력 크기 | 채널 |
|--------|-----------|-----------|------|
| Conv2D(stride=2) | 64×64 | 32×32 | 4→32 |
| Conv2D(stride=2) | 32×32 | 16×16 | 32→64 |
| Conv2D(stride=2) | 16×16 | 8×8 | 64→128 |
| Conv2D(stride=2) | 8×8 | 4×4 | 128→256 |
| Flatten | 4×4×256 | 4096 | — |
| Dense(μ) | 4096 | 128 | — |
| Dense(log σ²) | 4096 | 128 | — |

각 Conv2D 뒤에: BatchNorm → LeakyReLU(0.2)

## 디코더 레이어 구성

| 레이어 | 입력 크기 | 출력 크기 | 채널 |
|--------|-----------|-----------|------|
| Dense + Reshape | 128 | 4×4 | →256 |
| ConvTranspose2D(stride=2) | 4×4 | 8×8 | 256→128 |
| ConvTranspose2D(stride=2) | 8×8 | 16×16 | 128→64 |
| ConvTranspose2D(stride=2) | 16×16 | 32×32 | 64→32 |
| ConvTranspose2D(stride=2) | 32×32 | 64×64 | 32→4 |

마지막 레이어를 제외한 ConvTranspose2D 뒤에: BatchNorm → ReLU  
마지막 레이어: Sigmoid

## 손실 함수

```
L = Recon + β × KL

Recon = Σ(x - x̂)² / batch_size       (픽셀 단위 MSE 합산 후 배치 평균)
KL    = -0.5 × Σ(1 + log σ² - μ² - σ²) / batch_size
```

### 손실 정규화 방식

`reduction='sum' / batch_size` 를 사용합니다.  
`reduction='mean'` (픽셀 전체 평균)을 쓰면 Recon:KL 규모 불균형(~64배)으로 **posterior collapse** 발생 — 인코더가 입력을 무시하고 항상 μ≈0, σ≈1 출력.

## TF vs PyTorch 구현 차이

| 항목 | TF (04) | PyTorch (04-1) |
|------|---------|----------------|
| 텐서 포맷 | NHWC (B,H,W,C) | NCHW (B,C,H,W) |
| Padding | `padding='same'` | `padding=1` |
| BatchNorm momentum | 0.9 (PyTorch 동일하게 맞춤) | 0.1 (기본값) |
| 가중치 초기화 | Glorot(Xavier) uniform | Kaiming(He) uniform |
| LR 업데이트 | 스텝마다 (CosineDecay) | 에폭마다 (CosineAnnealingLR) |
| 그래디언트 클리핑 | `clip_by_global_norm(1.0)` | `clip_grad_norm_(1.0)` |
| Early Stopping | Keras 콜백 | 수동 카운터 |

## β-VAE

β > 1.0 으로 KL 항에 가중치를 높이면 잠재 공간 disentanglement 강화.

| β | 특성 |
|---|------|
| 1.0 | 표준 VAE — 재구성 품질 우선 |
| 2–4 | 재구성 품질과 분리도 균형 |
| 8–16 | 잠재 공간 분리 강화, 재구성 품질 저하 |

`05_beta_vae_experiment.ipynb`에서 β 목록을 지정해 비교 실험 가능.

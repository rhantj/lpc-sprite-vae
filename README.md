# LPC Sprite VAE

LPC(Liberated Pixel Cup) 64×64 RGBA 스프라이트 프레임을 학습하는 VAE 기반 이미지 생성 프로젝트.

## 노트북 실행 순서

| 번호 | 파일 | 역할 |
|------|------|------|
| 01 | `01_dataset_analysis.ipynb` | 원본 데이터셋 탐색 및 통계 |
| 02 | `02_preprocessing.ipynb` | 전처리 및 64×64 PNG 변환 |
| 03 | `03_data_restructure.ipynb` | 카테고리 재분류 (wings/tail 분리, zombie/skeleton 삭제, wound/wheelchair/prosthesis 추출) |
| 03-1 | `03-1_augment_body.ipynb` | body 데이터 좌우반전 증강 (2,272 → 4,544장) |
| 04 | `04_vae_model_tf.ipynb` | VAE 학습 — TensorFlow (주력) |
| 04-1 | `04-1_vae_model_pytorch.ipynb` | VAE 학습 — PyTorch (비교용) |
| 05 | `05_beta_vae_experiment.ipynb` | β-VAE 실험 (β=1~16 비교) |

> 03 → 03-1 은 데이터 준비 단계이므로 04를 실행하기 전에 반드시 완료해야 합니다.

## 데이터셋 구조

```
dataset/
└── processed/          # 카테고리별 64×64 RGBA PNG
    ├── body/           # 4,544 (원본 2,272 + 좌우반전 2,272)
    ├── wings/          # 100,264
    ├── tail/           # 93,593
    ├── wound/          # 2,528
    ├── wheelchair/     # 70
    ├── prosthesis/     # 532
    └── ...             # 기타 20개 카테고리 (총 26개)
```

총 프레임 수: 약 2,848,904장

## 체크포인트 구조

```
checkpoints_tf/         # TF VAE 모델
├── best.weights.h5
├── final.weights.h5
├── decoder/            # SavedModel (Streamlit 추론용)
└── config.json

checkpoints_pt/         # PyTorch VAE 모델
├── best.pt
├── final.pt
└── config.json

checkpoints_beta/       # β-VAE 실험 결과
├── beta_1.0.pt
├── beta_2.0.pt
├── beta_4.0.pt
├── beta_8.0.pt
├── best.pt             # 선택한 최적 β 모델
└── config.json
```

## 환경

- Python 3.10+
- TensorFlow 2.16+ (Metal GPU — Apple M1/M2)
- PyTorch 2.x (MPS 백엔드)
- Pillow, NumPy, Matplotlib

## 주요 하이퍼파라미터 (기본값)

| 파라미터 | 값 |
|----------|-----|
| 이미지 크기 | 64×64 RGBA |
| latent_dim | 128 |
| base_channels | 32 |
| epochs | 50 |
| batch_size | 128 |
| learning_rate | 0.01 (CosineAnnealing, min=1e-5) |
| beta | 1.0 (표준 VAE) |
| early stopping patience | epochs // 10 |

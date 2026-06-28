# LPC Sprite VAE

LPC(Liberated Pixel Cup) 64×64 RGBA 스프라이트 프레임을 학습하는 VAE 기반 이미지 생성 프로젝트.

## 노트북 실행 순서

| 번호 | 파일 | 역할 |
|------|------|------|
| 01 | `01_dataset_analysis.ipynb` | 원본 데이터셋 탐색 및 통계 |
| 02 | `02_preprocessing.ipynb` | 전처리 및 64×64 PNG 변환 |
| 03 | `03_data_restructure.ipynb` | 카테고리 재분류 (wings/tail 분리, zombie/skeleton 삭제 등) |
| 03-1 | `03-1_augment_body.ipynb` | body 데이터 좌우반전 증강 (2,272 → 4,544장) |
| 03-2 | `03-2_keyword_split.ipynb` | 카테고리 내 파일을 첫 키워드 서브폴더로 분리 (body 제외) |
| 04 | `04_vae_model_tf.ipynb` | VAE 학습 — TensorFlow |
| 04-1 | `04-1_vae_model_pytorch.ipynb` | VAE 학습 — PyTorch |
| 05 | `05_beta_vae_experiment.ipynb` | β-VAE 실험 (β=0.5/1.0/2.0/4.0 비교, β=1.0 최종 선택) |
| 06 | `06_latent_arithmetic_tf.ipynb` | Latent space arithmetic — TensorFlow |
| 06-1 | `06-1_latent_arithmetic_pytorch.ipynb` | Latent space arithmetic — PyTorch |
| 07 | `07_latent_visualization.ipynb` | VAE 잠재 공간 시각화 (PCA / t-SNE / UMAP) |
| 07-1 | `07-1_latent_visualization_cvae.ipynb` | CVAE 잠재 공간 시각화 (VAE와 비교) |
| 08 | `08_cvae_tf.ipynb` | CVAE 학습 — TensorFlow (β=1.0) |
| 08-1 | `08-1_cvae_pytorch.ipynb` | CVAE 학습 — PyTorch (β=1.5) |
| 09 | `09_cvae_perceptual_tf.ipynb` | CVAE + Perceptual Loss 파인튜닝 — TensorFlow |
| 09-1 | `09-1_cvae_perceptual_pytorch.ipynb` | CVAE + Perceptual Loss 파인튜닝 — PyTorch **★ 최종 모델** |

> 데이터 준비: 03 → 03-1 → 03-2 순서로 실행 후 04 진행  
> 09-1 PyTorch 버전이 조건부 생성 품질이 가장 우수해 Streamlit 앱에 사용합니다.

## 데이터셋 구조

```
dataset/
└── processed/          # 카테고리별 64×64 RGBA PNG
    ├── body/           # 4,544 (flat, 좌우반전 포함)
    ├── torso/
    │   ├── clothes/    # 가장 큰 서브셋
    │   ├── armour/
    │   └── ...         # chainmail, jacket, bandage, aprons, waist
    ├── legs/           # pants, skirts, armour, hose, leggings ...
    ├── feet/           # boots, shoes, sandals, socks ...
    ├── hair/           # long, afro, braid, ponytail ... (40+ 종)
    ├── hat/            # cloth, helmet, formal, headband ...
    ├── wings/          # 100,264 (단일 키워드, flat)
    ├── tail/           # 93,593 (단일 키워드, flat)
    └── ...             # 총 26개 카테고리
```

총 프레임 수: 약 2,848,904장

## 체크포인트 구조

```
checkpoints_tf/           # TF VAE (β=1.0)
checkpoints_pt/           # PyTorch VAE (β=1.0)
checkpoints_beta/         # β-VAE 실험 결과
checkpoints_cvae_tf/      # TF CVAE (β=1.0)
checkpoints_cvae_pt/      # PyTorch CVAE (β=1.5)
checkpoints_cvae_pl_tf/   # TF CVAE + Perceptual Loss
checkpoints_cvae_pl_pt/   # PyTorch CVAE + Perceptual Loss ★ Streamlit 사용
```

> 모든 체크포인트는 `.gitignore`로 제외됩니다. 로컬에서 각 노트북을 실행해 생성하세요.

## 환경

- Python 3.10+
- TensorFlow 2.16+ (Metal GPU — Apple M1/M2)
- PyTorch 2.x (MPS 백엔드)
- scikit-learn (PCA, t-SNE)
- umap-learn (UMAP 시각화, 선택)
- Pillow, NumPy, Matplotlib

## 주요 하이퍼파라미터

| 파라미터 | VAE | CVAE | CVAE + PL |
|----------|-----|------|-----------|
| latent_dim | 128 | 128 | 128 |
| base_channels | 32 | 32 | 32 |
| epochs | 50 | 100 | 50 (파인튜닝) |
| batch_size | 64 | 128 | 128 |
| learning_rate | 1e-3 | 1e-3 | 1e-4 |
| beta | 1.0 | 1.5 (PT) | 1.5 (PT) |
| lambda_perc | — | — | 0.00005 |

<div align="center">

# ST-HTM: Spatio-Temporal Hybrid Transformer-Mamba

**Gradually Unfrozen DINOv2 · Bidirectional Temporal Mamba · Multi-Scale SegFormer**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch%202.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA%2013.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/DINOv2%20ViT--S%2F14-0064E0?style=for-the-badge&logo=meta&logoColor=white" alt="DINOv2"/>
  <img src="https://img.shields.io/badge/BiMamba%20SSM-7C3AED?style=for-the-badge&logo=lightning&logoColor=white" alt="BiMamba"/>
  <img src="https://img.shields.io/badge/SegFormer%20Head-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white" alt="SegFormer"/>
</p>

<p align="center">
  <a href="results/report.md"><img src="https://img.shields.io/badge/Val%20mIoU%20·%2084.22%25-10B981?style=flat-square" alt="Val mIoU"/></a>
  <a href="results/report.md"><img src="https://img.shields.io/badge/Tool%20IoU%20·%2091.02%25-06B6D4?style=flat-square" alt="Tool IoU"/></a>
  <a href="results/report.md"><img src="https://img.shields.io/badge/Unseen%20mIoU%20·%2091.58%25-3B82F6?style=flat-square" alt="Unseen mIoU"/></a>
  <a href="results/hardware.png"><img src="https://img.shields.io/badge/154.7%20FPS%20(T4)-F97316?style=flat-square" alt="Throughput"/></a>
  <a href="results/metrics.json"><img src="https://img.shields.io/badge/29.21M%20Params-A855F7?style=flat-square" alt="Params"/></a>
  <a href="https://huggingface.co/datasets/tyluan/Endovis2017"><img src="https://img.shields.io/badge/EndoVis%202017-EC4899?style=flat-square" alt="Dataset"/></a>
</p>
</div>

---

**ST-HTM** is a fast, occlusion-aware video segmentation model built for robotic surgery on the **MICCAI EndoVis 2017** dataset. It combines a **gradually unfrozen DINOv2 ViT-S/14** backbone, **bidirectional Mamba** layers for linear-time temporal tracking, a lightweight spatial Transformer, and a **SegFormer All-MLP decoder** to segment 8 surgical instrument classes cleanly across video frames.

---

## Visual Showcase

#### Clean Unseen Video Prediction (`seq10.mp4`)

https://github.com/user-attachments/assets/c678d57d-d7a4-404e-8ea0-220c84765402

#### Simulated Multi-Frame Occlusion Test (`seq10_occlusion.mp4`)

https://github.com/user-attachments/assets/e99d59dd-03a4-4adf-9a3e-fea554edeaf0

<details>
<summary><b>View Keyframe Stills (Input | ST-HTM Prediction | Expert Ground Truth)</b></summary>
<br>
<p align="center">
  <img width="48%" alt="seq10_stills" src="https://github.com/user-attachments/assets/c7736e0a-e0ad-4077-a79b-0f62082ec50a" />
  <img width="48%" alt="seq10_occlusion_stills" src="https://github.com/user-attachments/assets/aac795fc-6851-4b24-8ab0-f594d8dd6880" />
</p>
</details>

---

## How It Works

Surgical video is hard for standard vision models: metallic tools reflect light, blood and tissue move constantly, and the camera shifts quickly. Frame-by-frame models flicker between frames, while full spatio-temporal Transformers run out of memory on long clips.

ST-HTM solves this with four simple ideas:

- **Clean, Video-Ordered Caching (`prepare_cache.py`):** Downloads `tyluan/Endovis2017`, rebuilds the true video frame order, applies one fixed crop per video to remove black borders, cleans noisy mask edges (`IGNORE = 255`), and saves `448×448` `uint8` tensors to `data/frames/`.
- **Gradual Backbone Unfreezing (`model.py`, `train.py`):** Starts with DINOv2 frozen so the new head can stabilize, then unfreezes 3 Transformer blocks at a time (epochs `4, 7, 10, 13`, and embeddings at `16`) using layer-wise learning rate decay (`0.8^depth`).
- **Linear-Time Temporal Tracking (`BiTemporalMamba`):** Runs forward and backward 2-layer Mamba SSMs (`d_state = 16`) per spatial patch across time in $O(T)$ time, with a learned `mask_token` for hidden or occluded frames.
- **Multi-Scale SegFormer Decoding:** Pulls features from 4 DINOv2 blocks `(2, 5, 8, 11)`, refines the deepest level with BiMamba and a 6-head spatial Transformer, builds a 4-level feature pyramid, and decodes masks with `SegformerDecodeHead`.

---

## Architecture

```mermaid
graph TD
    A["Raw Video Clips<br>[B, T, 3, 224, 224]"] --> B("GPU Coherent Augmentations<br>Affine Warp + Color Jitter + Frame Dropout")
    B --> C["DINOv2 ViT-S/14 Backbone<br>Gradually Unfrozen · Layers (2, 5, 8, 11)"]

    C -->|"Blocks 2, 5, 8<br>[B*T, 3, 384, 16, 16]"| P_SKIP["Multi-Scale Pyramid Adapters<br>ConvTranspose2d (strides 3.5, 7, 14)"]

    subgraph ST_CORE["Spatio-Temporal Core (Deepest Level: Block 11)"]
        C -->|"Block 11<br>[B, T, 256, 384]"| D["LayerNorm + Mask Token + Time Emb"]
        D --> E["Per-Patch Temporal Reshape<br>[B*256, T, 384]"]
        E --> F{"Bidirectional Mamba SSM<br>2-Layer Fwd + 2-Layer Bwd (d_state=16)"}
        F --> G["Residual Linear Fusion (768 → 384)"]
        G --> H["Spatial Transformer Layer<br>6-Head MHSA · [B*T, 256, 384]"]
    end

    subgraph DEC_HEAD["SegFormer All-MLP Decode Head"]
        P_SKIP --> I["4-Level Feature Pyramid<br>Channels: [96, 192, 384, 384]"]
        H -->|"Reshape [B*T, 384, 16, 16]"| I
        I --> J("SegformerDecodeHead<br>Hidden Dim = 256 · Dropout = 0.1")
        J --> K("Bilinear Upsample<br>[B, T, 8, 224, 224]")
    end

    K --> OUT["Multi-Class Instrument Masks (8 Classes)"]

    Loss["Cross-Entropy + Foreground Focal Tversky + Soft Dice<br>(FP32 · Ignore Index = 255)"] -.->|End-to-End Gradients| DEC_HEAD
    Loss -.->|LLRD 0.8^depth| ST_CORE

    classDef coralHighlight fill:#F45B69,stroke:#0B2027,stroke-width:2px,color:#FFFFFF;
    classDef darkBlock fill:#0B2027,stroke:#70A9A1,stroke-width:2px,color:#F6F1D1;
    classDef tealBlock fill:#40798C,stroke:#0B2027,stroke-width:1.5px,color:#F6F1D1;
    classDef aquaBlock fill:#70A9A1,stroke:#0B2027,stroke-width:1.5px,color:#0B2027;
    classDef lossConstraint fill:#0B2027,stroke:#40798C,stroke-width:1.5px,stroke-dasharray: 4 4,color:#F6F1D1;

    class A,F coralHighlight;
    class OUT darkBlock;
    class B,C,J,K tealBlock;
    class D,E,G,H,I,P_SKIP aquaBlock;
    class Loss lossConstraint;

    style ST_CORE fill:#0B2027,stroke:#40798C,stroke-width:2px,color:#F6F1D1
    style DEC_HEAD fill:#0B2027,stroke:#70A9A1,stroke-width:2px,color:#F6F1D1
    linkStyle default stroke:#40798C,stroke-width:1.5px;
```

### Built With

<div align="center">

| Component | Tools & Libraries | Role |
| :---: | :---: | :---: |
| **Backbone** | `dinov2_vits14` (`torch.hub`) | Extracts multi-layer visual features (blocks `2, 5, 8, 11`) with gradual unfreezing |
| **Temporal Engine** | `MambaModel` (`transformers`), optional `mamba_ssm` & `causal_conv1d` | Forward & backward per-patch SSMs (`d_state=16`) with PyTorch reference fallback |
| **Decoder** | `nn.TransformerEncoderLayer`, `SegformerDecodeHead` | Spatial refinement and 4-scale pyramid fusion into 8-class segmentation logits |
| **Loss & Optimizer** | Custom FP32 `FocalTverskyDiceLoss`, Fused `AdamW`, Cosine Warmup | Handles extreme class imbalance while ignoring noisy mask edges (`255`) |
| **Training** | `torch.distributed` (DDP), `SyncBatchNorm`, `torch.amp` (`FP16`/`BF16`) | Dual-T4 / A100 training with coherent GPU-side clip augmentations (`F.affine_grid`) |
| **Data & Video** | Hugging Face `datasets`, `OpenCV`, `NumPy`, `imageio[ffmpeg]`, `Matplotlib` | Dataset caching, mask cleanup, profiling plots, and side-by-side MP4 rendering |

</div>

---

## Validation Results (EndoVis 2017)

Tested on the full validation split (`901` frames across `10` sequences, upsampled to `448×448` before `argmax`) using the **Epoch 15** checkpoint (`best_st_htm.pt`, `224×224` input). Multi-class means are averaged over tool classes present in the ground truth (`support > 0`).

### Overall Performance

<div align="center">

| Split | mIoU (%) | mDice (%) | mPrecision (%) | mRecall (%) | Pixel Acc (%) | Binary Tool IoU (%) | Binary Tool Dice (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Validation (Seq 1–10)** | **84.22%** | **91.05%** | 90.68% | 92.58% | 98.30% | **91.02%** | **95.30%** |
| **Seen Videos (Seq 1–8)** | 84.48% | 91.23% | 90.76% | 92.79% | 97.76% | 89.36% | 94.38% |
| **Unseen Videos (Seq 9–10)** | **91.58%** | **95.57%** | 93.42% | 97.99% | 99.36% | **95.13%** | **97.50%** |

</div>
> *Note:* Sequences 9 (1 frame) and 10 (300 frames, `large_needle_driver`) are completely unseen surgical videos during training.
<p align="center"><img width="2080" height="546" alt="training_curves" src="https://github.com/user-attachments/assets/7e87b8b2-79d5-4d95-b8e2-262817fdfcec" />
</p>
### Per-Class Breakdown

<div align="center">

| Instrument Class | IoU (%) | Dice (%) | Precision (%) | Recall (%) | Specificity (%) | GT Pixels |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `background` | 98.57% | 99.28% | 99.56% | 99.01% | 97.05% | 154,896,099 |
| `large_needle_driver` | **95.51%** | **97.70%** | 98.21% | 97.20% | 99.86% | 13,191,949 |
| `monopolar_curved_scissors` | **93.77%** | **96.79%** | 95.16% | 98.47% | 99.97% | 1,182,014 |
| `vessel_sealer` | **85.71%** | **92.31%** | 96.32% | 88.61% | 99.96% | 2,281,336 |
| `bipolar_forceps` | **80.00%** | **88.89%** | 95.48% | 83.15% | 99.90% | 4,499,395 |
| `prograsp_forceps` | 66.09% | 79.58% | 68.22% | 95.49% | 99.49% | 2,036,336 |
| `grasping_retractor` / `other` | — | — | — | — | 99.99% / 99.50% | 0 *(absent in val)* |

</div>
<p align="center"><img width="1820" height="624" alt="per_class_metrics" src="https://github.com/user-attachments/assets/fac8ca9b-0076-4ef8-913c-2ed8198b7340" /></p>
<p align="center"><img width="1105" height="975" alt="confusion_matrix" src="https://github.com/user-attachments/assets/462a5c16-967f-4947-bec3-2640174c5d1a" /></p>

### Temporal Stability (Per-Sequence Highlights)

Prediction flicker (the share of pixels changing class between consecutive frames) stays closely aligned with ground-truth motion:

<div align="center">

| Sequence | Split | Frames | mIoU (%) | Tool IoU (%) | Pred Flicker (%) | GT Flicker (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seq 3** | Seen | 75 | 95.70% | 95.70% | 12.50% | 12.37% |
| **Seq 4** | Seen | 75 | 94.21% | 95.25% | 7.86% | 7.62% |
| **Seq 5** | Seen | 75 | 92.07% | 92.68% | 11.84% | 11.58% |
| **Seq 8** | Seen | 75 | 95.66% | 95.87% | 11.93% | 11.80% |
| **Seq 10** | **Unseen** | 300 | **95.25%** | **95.13%** | **10.10%** | **10.17%** |

</div>
<p align="center"><img width="1300" height="520" alt="per_sequence" src="https://github.com/user-attachments/assets/511b5b0b-04ca-49c1-9c64-c580b482a1b4" /></p>
<p align="center"><img width="1560" height="2054" alt="frame_timeline" src="https://github.com/user-attachments/assets/20e55421-5d65-43fc-be70-b64435a8a23b" /></p>

### Occlusion Stress Test

Consecutive frames ($k$) are blacked out and replaced with the learned `mask_token`, then scored strictly on the hidden frames:

<div align="center">

| Hidden Frames ($k$) | Hidden-Frame mIoU (%) | Hidden-Frame Tool IoU (%) |
| :---: | :---: | :---: |
| **0 (Clean Baseline)** | 60.15% *(84.22% GT-present)* | **91.02%** |
| **1 Frame Hidden** | 39.87% | **48.63%** |
| **2 Frames Hidden** | 36.25% | **41.91%** |
| **3 Frames Hidden** | 30.85% | **33.00%** |
| **5 Frames Hidden** | 20.06% | **21.02%** |
| **8 Frames Hidden** | 10.54% | **7.98%** |
| **10 Frames Hidden** | 9.94% | **9.27%** |

</div>
<p align="center"><img width="910" height="520" alt="occlusion_robustness" src="https://github.com/user-attachments/assets/517a9c31-24a1-4998-a202-3bce5238b26b" /></p>

---

## Hardware & Scaling Benchmark

Measured on a single **NVIDIA Tesla T4** (`PyTorch 2.11.0+cu130`, `FP16`, `224×224` input, unfused PyTorch Mamba scan).

- **Model Size:** `29.21M` parameters (`22.06M` DINOv2 + `7.15M` temporal/SegFormer head) · `111.4 MB` FP32
- **Compute:** `18.40 GFLOPs / frame` (`12.25` DINOv2 + `6.15` head)
- **Online Streaming (1 new frame + cached 16-frame window):** `97.0 ms / frame` (`8.45 ms` DINOv2 + `88.57 ms` unfused head) = **10.3 FPS** at **856 MB** peak VRAM

### Clip Length Scaling (Batch Size = 1)

<div align="center">

| Clip Length ($T$) | p50 Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **4 frames** | 29.8 ms | 31.4 ms | 133.6 FPS | 384 MB |
| **8 frames** | 55.1 ms | 55.5 ms | 145.2 FPS | 607 MB |
| **16 frames** | 111.5 ms | 112.4 ms | 143.6 FPS | 891 MB |
| **32 frames** | 228.0 ms | 232.9 ms | 139.9 FPS | 1,620 MB |
| **64 frames** | 479.1 ms | 484.6 ms | 133.5 FPS | 3,077 MB |

</div>

### Batched Throughput ($T = 16$)

<div align="center">

| Clips / Batch | p50 Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 105.7 ms | 106.5 ms | 151.3 FPS | 891 MB |
| **2** | 206.7 ms | 209.2 ms | **154.7 FPS** | 1,940 MB |
| **4** | 417.7 ms | 420.6 ms | 153.2 FPS | 3,717 MB |
| **8** | 866.5 ms | 871.8 ms | 148.0 FPS | 7,273 MB |

</div>
<p align="center"><img width="2080" height="546" alt="hardware" src="https://github.com/user-attachments/assets/35952f23-b73b-45c7-a146-24915ff612a7" /></p>

---

## Loss Function

Computed in `FP32` over valid pixels (`target != 255`) in `loss.py`, combining **Cross-Entropy**, **Foreground Focal Tversky**, and **Soft Dice**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FT}} + \mathcal{L}_{\text{Dice}} + \lambda_{\text{CE}}\,\mathcal{L}_{\text{CE}}$$

$$\mathcal{L}_{\text{FT}} = \frac{1}{\vert{}C_{\text{fg}}\vert{}} \sum_{c \in C_{\text{fg}}} \left( 1 - \frac{\text{TP}_c + s}{\text{TP}_c + \alpha\,\text{FN}_c + \beta\,\text{FP}_c + s} \right)^{\gamma}$$

- $\alpha = 0.7,\ \beta = 0.3$: Penalizes missed tool pixels more heavily so thin shafts and jaws aren't lost.
- $\gamma = 0.75,\ \lambda_{\text{CE}} = 1.0,\ s = 1.0$: Keeps focus on hard boundaries while stabilizing probabilities.

---

## Repository Structure

```text
ST-HTM/
├── install_kernels.py      # Optional prebuilt CUDA wheel installer (causal_conv1d & mamba_ssm)
├── prepare_cache.py        # Downloads EndoVis 2017, cleans masks, crops borders, caches 448px .pt files
├── model.py                # DinoBackbone (gradual unfreeze), BiTemporalMamba, Spatial Transformer & SegFormer head
├── loss.py                 # FP32 FocalTverskyDiceLoss (CE + Focal Tversky + Soft Dice)
├── train.py                # Single/Multi-GPU DDP training, GPU augmentations, LLRD, auto-resume & early stopping
├── infer.py                # AMP helpers, checkpoint loader & sliding-window predict_sequence()
├── test.py                 # Full evaluation: accuracy, occlusion sweep, flicker stability, FLOPs/VRAM & plots
├── make_video.py           # Side-by-side MP4 renderer (input | prediction | ground truth + occlusion demo)
├── checkpoints/            # best_st_htm.pt, last.pt, history.json
└── results/                # report.md, metrics.json, per_class.csv, plots & videos/
```

---

## Quickstart

```bash
# 1. Install dependencies & optional fused Mamba CUDA kernels
pip install -q transformers datasets opencv-python-headless "imageio[ffmpeg]" matplotlib
python install_kernels.py

# 2. Download EndoVis 2017 and build sequence-ordered 448px cache
python prepare_cache.py

# 3. Train (single GPU or multi-GPU DDP; auto-resumes from checkpoints/last.pt)
torchrun --standalone --nproc_per_node=2 train.py

# 4. Run full evaluation, occlusion stress test, hardware benchmark & plot generation
python test.py

# 5. Render showcase videos (clean + simulated occlusion)
python make_video.py --seqs 10
python make_video.py --seqs 10 --occlusion-demo
```

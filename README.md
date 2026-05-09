# 🇻🇳 Vietnamese Sports News Summarization — Fine-tuning Pipeline

> **Môn học**: Xử lý Ngôn ngữ Tự nhiên (NLP)  
> **Sinh viên**: 52300102 & 52300107  
> **Nền tảng**: Google Colab (GPU T4)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu môi trường](#-yêu-cầu-môi-trường)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Hướng dẫn chạy từng bước](#-hướng-dẫn-chạy-từng-bước)
- [Cấu hình Hyperparameters](#-cấu-hình-hyperparameters)
- [Metrics đánh giá](#-metrics-đánh-giá)
- [Kết quả & Visualizations](#-kết-quả--visualizations)
- [Demo Gradio](#-demo-gradio)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Lưu ý & Xử lý lỗi](#-lưu-ý--xử-lý-lỗi)

---

## 📖 Giới thiệu

Dự án thực hiện **fine-tune ≥ 2 mô hình pretrained tiếng Việt** cho bài toán **tóm tắt văn bản thể thao** (nguồn VnExpress), với các ràng buộc nghiêm ngặt:

| Ràng buộc | Mô tả |
|---|---|
| **Độ dài output** | 20–25% so với văn bản gốc |
| **Bảo toàn số liệu** | 100% con số trong bản tóm tắt phải khớp với gốc |
| **Hai mô hình** | VietAI/vit5-base & vinai/bartpho-word |
| **Chiến lược training** | Two-Stage: XL-Sum (tổng quát) → VnExpress (chuyên biệt) |

### Mô hình sử dụng

| Mô hình | Hugging Face ID | Chiến lược | Tham số |
|---|---|---|---|
| **ViT5-base** | `VietAI/vit5-base` | Full fine-tune (Two-stage) | ~223M |
| **BARTpho-word** | `vinai/bartpho-word` | LoRA + Two-stage | ~396M (+LoRA) |

---

## 🏗️ Kiến trúc hệ thống

```
VnExpress Sports Dataset (CSV)
        │
        ▼
┌─────────────────────────────────────────┐
│         Two-Stage Training              │
│                                         │
│  Stage 1: XL-Sum Vietnamese (3K mẫu)   │  ← Học cách tóm tắt tổng quát
│         ↓                               │
│  Stage 2: VnExpress Sports Dataset      │  ← Domain-specific fine-tuning
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────┐    ┌────────────────────┐
│  ViT5-base   │    │  BARTpho-word      │
│  (Full FT)   │    │  (LoRA r=16, α=32) │
└──────────────┘    └────────────────────┘
        │                    │
        └────────┬───────────┘
                 ▼
    ┌─────────────────────────┐
    │  Length-Constrained     │
    │  Generation (20–25%)    │
    │  + Number Accuracy      │
    └─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │   Gradio Demo           │
    │   (Interactive UI)      │
    └─────────────────────────┘
```

---

## ⚙️ Yêu cầu môi trường

### Nền tảng
- **Google Colab** với **GPU T4** (bắt buộc — notebook được thiết kế cho Colab)
- RAM: ≥ 12 GB (T4 VRAM: 15 GB)
- Google Drive được mount

### Thư viện chính (tự động cài trong Cell 1)

```
transformers==4.41.2
accelerate==0.30.1
peft==0.11.1
sentence-transformers==3.0.1
datasets==2.19.1
evaluate
sacrebleu
rouge_score
bert_score
sentencepiece
gradio
nltk
plotly
pandas
py7zr
bitsandbytes
huggingface_hub
```

> ⚠️ **Quan trọng**: Phiên bản `transformers==4.41.2` và `datasets==2.19.1` đã được kiểm tra tương thích. **Không tự ý nâng cấp** để tránh lỗi `KeyError: 0` với sentencepiece.

---

## 📂 Chuẩn bị dữ liệu

### Cấu trúc thư mục trên Google Drive

Trước khi chạy notebook, hãy tạo cấu trúc sau trong **Google Drive**:

```
MyDrive/
└── nlp/
    ├── train.csv      ← Dữ liệu training VnExpress (thể thao)
    ├── val.csv        ← Dữ liệu validation VnExpress (thể thao)
    ├── test.csv       ← Dữ liệu test VnExpress (thể thao)
    └── outputs/       ← (Tự tạo khi chạy) chứa model checkpoints
```

### Định dạng file CSV

Mỗi file CSV phải có **đúng 2 cột** (tên cột không phân biệt hoa thường):

| Cột | Mô tả | Ví dụ |
|---|---|---|
| `content` (hoặc `article`) | Văn bản gốc (bài báo thể thao VnExpress) | "Đội tuyển Việt Nam vừa giành chiến thắng..." |
| `summary` | Bản tóm tắt mẫu | "ĐT Việt Nam thắng 2-0 trong trận đấu..." |

> 💡 Notebook tự động đổi tên cột `article` → `content` nếu cần.

### Nguồn dữ liệu XL-Sum (Stage 1)
Dataset `GEM/xlsum` (tiếng Việt) được **tự động tải** từ Hugging Face Hub — không cần chuẩn bị thủ công. Notebook lấy 3.000 mẫu train và 300 mẫu validation.

---

## 🚀 Hướng dẫn chạy từng bước

### Bước 0: Mở notebook trên Google Colab

1. Tải file `52300102_52300107.ipynb` lên **Google Drive**
2. Click chuột phải → **Mở bằng Google Colaboratory**
3. Vào **Runtime → Change runtime type → GPU → T4**
4. Xác nhận đã kết nối GPU (biểu tượng RAM/Disk góc trên phải)

---

### Bước 1: Cài đặt thư viện (Cell 1)

```python
# Chạy Cell 1 — mất khoảng 2-3 phút
!pip uninstall -y transformers peft accelerate sentence-transformers
!pip install -q transformers==4.41.2 accelerate==0.30.1 peft==0.11.1 ...
```

✅ **Kết quả mong đợi**: `✓ Cài đặt hoàn tất!`

> ⚠️ Nếu có thông báo **"Restart runtime"**, hãy restart và **bỏ qua Cell 1** (chạy từ Cell 2).

---

### Bước 2: Mount Google Drive & Import (Cell 2)

```python
# Chạy Cell 2
from google.colab import drive
drive.mount('/content/drive')
```

- Một cửa sổ pop-up sẽ xuất hiện yêu cầu **cấp quyền truy cập Drive**
- Chọn tài khoản Google có chứa thư mục `MyDrive/nlp/`

✅ **Kết quả mong đợi**:
```
  Device: cuda
   GPU: Tesla T4
   VRAM: 15.8 GB
✓ Import hoàn tất!
```

> 📁 Đảm bảo đường dẫn `DATA_DIR = '/content/drive/MyDrive/nlp'` khớp với cấu trúc Drive của bạn. Nếu khác, sửa biến này trong Cell 2.

---

### Bước 3: Load & Khám phá Dữ liệu (Cell 3)

```python
# Chạy Cell 3 — đọc train/val/test CSV
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
val_df   = pd.read_csv(f'{DATA_DIR}/val.csv')
test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')
```

✅ **Kết quả mong đợi**: Thống kê số mẫu và phân phối độ dài văn bản.

---

### Bước 4: Tiện ích Số liệu & Tăng cường Dữ liệu (Cell 4)

Định nghĩa:
- `extract_numbers()` — trích xuất số liệu bằng regex
- `number_accuracy()` — tính tỉ lệ số khớp giữa source và summary

Không cần tương tác, chỉ cần chạy cell.

---

### Bước 5: Cài đặt lại datasets & Load XL-Sum (Cell 5a & 5b)

**Cell 5a** — cài đúng phiên bản datasets:
```python
!pip uninstall -y datasets
!pip install -q datasets==2.19.1
```

> ⚠️ Sau khi chạy Cell 5a, **KHÔNG restart runtime**. Tiếp tục chạy Cell 5b ngay.

**Cell 5b** — tải XL-Sum từ Hugging Face:
```python
xlsum = load_dataset('GEM/xlsum', 'vietnamese', trust_remote_code=True)
# Lấy 3000 train + 300 val
```

✅ **Kết quả mong đợi**: Progress bar tải dataset, sau đó thống kê 3000/300 mẫu.

---

### Bước 6: Cấu hình Hyperparameters (Cell 6)

Cell này định nghĩa tất cả cấu hình (xem thêm phần [Cấu hình Hyperparameters](#-cấu-hình-hyperparameters)). Không cần tương tác, chỉ cần chạy.

✅ **Kết quả mong đợi**:
```
✓ Cấu hình hyperparameters hoàn tất!
  [vit5] VietAI/vit5-base | LoRA=False | BS=2x8 | LR_s2=0.0001
  [bartpho] vinai/bartpho-word | LoRA=True | BS=2x8 | LR_s2=5e-05
```

---

### Bước 7: Metrics Suite (Cell 7)

Khởi tạo các metrics: ROUGE, BLEU (sacrebleu), METEOR, BERTScore.

---

### Bước 8: Hàm Fine-tuning Chung (Cell 8)

Định nghĩa `LossLoggerCallback` để theo dõi loss theo epoch. Không cần tương tác.

---

### Bước 9: Fine-tune ViT5 — Two-Stage (Cell 9a & 9b)

```
Stage 1 (XL-Sum) → vit5_stage1/
Stage 2 (VnExpress Sports) → vit5_stage2/
Final model → vit5_final/
```

**Thời gian ước tính** (T4 GPU):
- Stage 1 (3 epochs, 3000 mẫu): ~25–35 phút
- Stage 2 (5 epochs, dataset thể thao VnExpress): phụ thuộc kích thước dataset

✅ **Checkpoint tự động**: Nếu Colab bị ngắt, chạy lại cell — notebook sẽ **tự detect checkpoint** và resume từ điểm cuối.

---

### Bước 10: Fine-tune BARTpho — LoRA + Two-Stage (Cell 10)

```
Stage 1 (XL-Sum + LoRA) → bartpho_stage1/
Stage 2 (VnExpress Sports + LoRA) → bartpho_stage2/
Final model → bartpho_final/
```

**Thời gian ước tính** (T4 GPU):
- Stage 1 (3 epochs): ~30–45 phút
- Stage 2 (5 epochs): phụ thuộc kích thước dataset

> 💡 BARTpho dùng LoRA (`r=16, alpha=32`) nên tiêu thụ ít VRAM hơn full fine-tune.

---

### Bước 11: Visualize Loss Curves (Cell 11a)

Vẽ đồ thị train/val loss cho cả 4 stages (2 mô hình × 2 stages).

---

### Bước 11b: Sinh Predictions & Evaluate (Cell 11b)

```python
TEST_CSV = '/content/drive/MyDrive/nlp/test.csv'
# Sinh tóm tắt theo batch và tính ROUGE/BLEU/METEOR/BERTScore
```

---

### Bước 12: Length-Constrained Generation (Cell 12)

Hàm `generate_summary()` với ràng buộc độ dài 20–25% và hậu xử lý số liệu.

---

### Bước 13: So sánh Metrics (Cell 14)

Vẽ biểu đồ so sánh toàn diện:
- Bar chart: ROUGE-1/2/L, BLEU, METEOR, BERTScore F1
- So sánh Number Accuracy & Length Compliance

---

### Bước 14: Demo Gradio (Cell 17)

Khởi chạy giao diện tương tác:

```python
# Load models từ disk (uncomment nếu cần):
# vit5_tokenizer = AutoTokenizer.from_pretrained(f'{OUTPUT_DIR}/vit5_final')
# vit5_model     = AutoModelForSeq2SeqLM.from_pretrained(f'{OUTPUT_DIR}/vit5_final')

# Chạy Gradio app
demo.launch(share=True)  # share=True để lấy public URL
```

✅ **Kết quả**: URL dạng `https://xxxx.gradio.live` — có thể mở trên bất kỳ thiết bị nào.

---

## ⚙️ Cấu hình Hyperparameters

### Thông số chung

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `MAX_INPUT_LEN` | 1024 | Token tối đa cho input |
| `MAX_TARGET_LEN` | 256 | Token tối đa cho output |
| `MIN_TARGET_LEN` | 50 | Minimum tokens sinh ra |
| `num_beams` | 4 | Beam search width |
| `length_penalty` | 1.2 | Khuyến khích output dài hơn |
| `no_repeat_ngram_size` | 3 | Tránh lặp 3-gram |
| `SEED` | 42 | Random seed cho reproducibility |

### ViT5-base

| Tham số | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 3e-4 | 1e-4 |
| Epochs | 3 | 5 |
| Batch size | 2 | 2 |
| Gradient accumulation | 8 | 8 |
| Effective batch size | **16** | **16** |
| LoRA | ❌ (full FT) | ❌ (full FT) |

### BARTpho-word

| Tham số | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 2e-4 | 5e-5 |
| Epochs | 3 | 5 |
| Batch size | 2 | 2 |
| Gradient accumulation | 8 | 8 |
| LoRA rank (r) | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.1 |

---

## 📊 Metrics đánh giá

| Metric | Thư viện | Mô tả |
|---|---|---|
| **ROUGE-1/2/L** | `evaluate` (rouge) | Recall-Oriented Understudy for Gisting Evaluation |
| **BLEU** | `sacrebleu` | Bilingual Evaluation Understudy |
| **METEOR** | `evaluate` (meteor) | Metric for Evaluation of Translation with Explicit ORdering |
| **BERTScore F1** | `bert_score` | Semantic similarity dùng BERT embeddings |
| **Number Accuracy** | Custom (regex) | % số liệu trong summary khớp với source |
| **Length Compliance** | Custom | % mẫu có độ dài 20–25% so với gốc |

---

## 📈 Kết quả & Visualizations

Sau khi chạy xong, các file sau được lưu vào `OUTPUT_DIR`:

```
outputs/
├── vit5_stage1/          ← Checkpoints Stage 1
├── vit5_stage2/          ← Checkpoints Stage 2
├── vit5_final/           ← Model ViT5 đã fine-tune hoàn chỉnh
├── bartpho_stage1/       ← Checkpoints Stage 1
├── bartpho_stage2/       ← Checkpoints Stage 2
├── bartpho_final/        ← Model BARTpho đã fine-tune hoàn chỉnh
├── loss_curves.png       ← Đồ thị train/val loss (4 stages)
└── metrics_comparison.png ← Biểu đồ so sánh metrics
```

---

## 🎮 Demo Gradio

Giao diện Gradio (Cell 17) hỗ trợ:

- **Nhập văn bản** tiếng Việt bất kỳ
- **Chọn mô hình**: ViT5 hoặc BARTpho (hoặc cả hai)
- **Hiển thị tóm tắt** kèm các metrics: ROUGE, Number Accuracy, Length Compliance
- **Dark-mode UI** tích hợp sẵn

---

## 📁 Cấu trúc thư mục Repository

```
52300102_52300107_Final_Report_NLP/
├── README.md                      ← File này
├── 52300102_52300107.ipynb        ← Notebook chính
└── Demo/                          ← (Trống) Để chứa video/ảnh demo
```

---

## ⚠️ Lưu ý & Xử lý lỗi

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `KeyError: 0` | Xung đột phiên bản transformers/sentencepiece | Dùng đúng `transformers==4.41.2` |
| `IndexError: piece id is out of range` | datasets không tương thích | Cài `datasets==2.19.1` |
| `CUDA out of memory` | VRAM không đủ | Giảm `batch_size` xuống 1, tăng `grad_accum` lên 16 |
| `FileNotFoundError: train.csv` | Sai đường dẫn Drive | Kiểm tra `DATA_DIR` trong Cell 2 |
| `Drive mount failed` | Session hết hạn | Chạy lại Cell 2, đăng nhập lại |

### Mẹo Resume Training

Notebook hỗ trợ **tự động resume** từ checkpoint cuối cùng:
```python
# Hàm load_latest_checkpoint() tự tìm checkpoint-XXX/ mới nhất
# Nếu Colab ngắt kết nối, chỉ cần:
# 1. Kết nối lại runtime
# 2. Chạy lại từ Cell 1 đến Cell 8
# 3. Chạy lại Cell 9 hoặc 10 — sẽ tự tiếp tục
```

### Tối ưu tốc độ

- **Dùng GPU T4** (không dùng CPU)
- BARTpho dùng LoRA → tiết kiệm VRAM, cho phép `batch_size=2` trên T4
- Giảm `N_XLSUM_TRAIN` xuống 1000–1500 nếu muốn Stage 1 nhanh hơn (có thể ảnh hưởng chất lượng nhỏ)

---

## 👥 Thông tin nhóm

| MSSV | Vai trò |
|---|---|
| **52300102** | Xây dựng pipeline, Fine-tuning ViT5 |
| **52300107** | Fine-tuning BARTpho, Đánh giá metrics, Demo Gradio |

---

*README được tạo ngày 09/05/2026*
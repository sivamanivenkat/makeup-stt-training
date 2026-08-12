# Colab setup — Moonshine fine-tuning

Paste each cell into Colab in order. Runtime → Change runtime type → GPU first.

## 1. Check GPU

```python
!nvidia-smi
```

## 2. Mount Drive (checkpoint backup + dataset storage)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Clone the repo

```bash
!git clone https://github.com/sivamanivenkat/makeup-stt-training.git
%cd makeup-stt-training
```

## 4. Get the dataset onto the Colab disk

`dataset/` (~9GB of audio) is gitignored, not in the repo. Two options:

**A — already zipped it to Drive:**
```bash
!unzip -q /content/drive/MyDrive/makeup-stt/dataset.zip -d /content/makeup-stt-training
```

**B — haven't uploaded it yet**, zip it locally first and upload to Drive, or upload directly from this machine:
```python
from google.colab import files
uploaded = files.upload()  # pick dataset.zip — slow for 9GB, prefer Drive
```

Either way, end state must be `./dataset/{train,val,test}/metadata.jsonl` + `audio/*.wav` sitting in the Colab working directory.

## 5. Install dependencies

```bash
!pip install -r requirements.txt
!pip install -r requirements-moonshine.txt
```

## 6. Verify the model class is importable (base doesn't need this, but check anyway)

```python
from transformers import MoonshineForConditionalGeneration
print("ok")
```

If targeting `moonshine-streaming-medium` instead of `moonshine-base`, also check:
```python
try:
    from transformers import MoonshineStreamingForConditionalGeneration
    print("streaming ok")
except ImportError:
    print("need: !pip install -U git+https://github.com/huggingface/transformers.git")
```

## 7. Run training

**Base model** (61.5M params, fits comfortably on any Colab GPU):
```bash
!python train_moonshine.py \
    --model UsefulSensors/moonshine-base \
    --dataset ./dataset \
    --output ./model_moonshine_base \
    --backup-dir /content/drive/MyDrive/makeup-stt/checkpoints-moonshine-base \
    --batch 8 --grad-accum 2 --epochs 3
```

**Streaming-medium** (245M params — lower batch on T4/16GB):
```bash
!python train_moonshine.py \
    --model UsefulSensors/moonshine-streaming-medium \
    --dataset ./dataset \
    --output ./model_moonshine_medium \
    --backup-dir /content/drive/MyDrive/makeup-stt/checkpoints-moonshine-medium \
    --batch 4 --grad-accum 4 --epochs 3
```

If the runtime disconnects mid-training, just rerun the same cell — checkpoints restore automatically from `--backup-dir`.

## 8. Grab the finished model

```bash
!cp -r ./model_moonshine_base/final /content/drive/MyDrive/makeup-stt/model_moonshine_base_final
```

Or zip and download directly:
```python
!zip -r final_model.zip ./model_moonshine_base/final
from google.colab import files
files.download('final_model.zip')
```

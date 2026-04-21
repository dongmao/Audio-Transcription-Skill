# 🎙️ Audio Transcription Skill

> 音频转录精炼全流程——从下载到精炼，一键搞定

一键完成音频的**下载 → 语音转文字 → 精炼版整理**全流程。专为中文场景优化，支持 B站视频、网盘分享、HTTP 直链等多种来源。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🌐 **全来源覆盖** | B站视频、极空间/百度/阿里云盘分享、HTTP直链、本地文件 |
| 🧠 **Qwen3-ASR 转录** | 中文识别精度最高，1.7B 小模型 GPU 高效推理 |
| 📝 **精炼版整理** | 按主题重组 + 口语清理 + 信息分层 + 关键数据速览 |
| 🔗 **智能重叠去重** | 分片尾部 5s 重叠，避免句子断裂，自动拼接 |
| ♻️ **断点续传** | 长音频中断后可继续，不重复劳动 |
| ⚡ **全平台 GPU** | Intel XPU (Arc) / NVIDIA CUDA / Apple MPS，自动检测 |
| 🇨🇳 **国内镜像** | ModelScope 下载模型，阿里云 PyPI 镜像安装依赖 |
| ✅ **完整性验证** | 自动检查覆盖率、失败率，确保无遗漏 |

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 获取音频                                            │
│  B站 → yt-dlp | 网盘 → Playwright+Edge | 直链 → curl       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 语音转文字（Qwen3-ASR-1.7B, 必须 GPU）              │
│  智能分片(30s+5s重叠) → 批量推理 → 断点续传 → 重叠去重拼接   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 生成精炼版（AI 整理）                                │
│  话题分章 → 口语清理 → 信息分层 → 表格/列表 → 完整性验证     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- GPU（Intel Arc / NVIDIA CUDA / Apple MPS，**禁止纯 CPU 推理**）
- ffmpeg
- Microsoft Edge 浏览器（网盘下载用，macOS 使用系统浏览器）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/dongmao/Audio-Transcription-Skill.git
cd Audio-Transcription-Skill

# 2. 安装 Python 依赖（阿里云镜像）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. Intel Arc GPU 用户额外安装 IPEX
pip install intel-extension-for-pytorch -i https://mirrors.aliyun.com/pypi/simple/

# 4. NVIDIA GPU 用户安装 CUDA 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. 安装 ffmpeg（Windows）
winget install ffmpeg

# 6. 安装 Playwright 浏览器支持
python -m playwright install  # 使用系统 Edge 无需此步
```

### 验证环境

```bash
# 检查 GPU
python -c "import torch; print('XPU:', hasattr(torch,'xpu') and torch.xpu.is_available()); print('CUDA:', torch.cuda.is_available())"

# 检查 qwen-asr
python -c "from qwen_asr import Qwen3ASRModel; print('qwen-asr OK')"

# 检查 Playwright
python -c "from playwright.sync_api import sync_playwright; print('playwright OK')"
```

## 📖 使用方法

### Step 1: 下载音频

```bash
# B站视频
python download_audio.py "https://www.bilibili.com/video/BV1xwQtBjEN1" -o "output.mp3"

# 极空间/百度网盘分享（自动启动 Edge 浏览器）
python download_audio.py "https://t3.znas.cn/xxxxx" -o "output.mp3"

# HTTP 直链
python download_audio.py "https://example.com/audio.mp3" -o "output.mp3"

# 本地文件（直接跳过下载步骤）
```

### Step 2: 语音转文字

```bash
# 基本用法（自动检测 GPU）
python transcribe_qwen3_asr.py "output.mp3"

# 指定 Intel XPU
python transcribe_qwen3_asr.py "output.mp3" -d "xpu:0"

# 指定 NVIDIA CUDA
python transcribe_qwen3_asr.py "output.mp3" -d "cuda:0"

# 指定 Apple MPS
python transcribe_qwen3_asr.py "output.mp3" -d "mps"

# 自定义分片和输出
python transcribe_qwen3_asr.py "output.mp3" -s 25 --overlap 3 -o "转录结果.txt"
```

**转录参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output, -o` | 输出文件路径 | 与音频同名 `_Qwen3ASR.txt` |
| `--segment-len, -s` | 分片长度（秒） | 30 |
| `--overlap` | 重叠长度（秒） | 5 |
| `--device, -d` | 推理设备 | 自动检测（GPU优先） |
| `--batch-size, -b` | 批次大小 | 自动（根据GPU显存） |
| `--model-id` | 模型 ID | `Qwen/Qwen3-ASR-1.7B` |
| `--no-overlap-dedup` | 禁用重叠去重 | 关 |
| `--language` | 识别语言 | `Chinese` |
| `--max-new-tokens` | 最大生成 token 数 | 2048 |

**性能参考：**

| 设备 | 音频时长 | 转录耗时 | 实时率 |
|------|---------|---------|--------|
| Intel Arc 140V (16GB) | 59min | 3.1min | ~19x |
| NVIDIA RTX 4090 | 59min | ~3min | ~20x |
| Apple M1/M2/M3 (MPS) | 59min | ~8-12min | ~5-7x |
| CPU (i7) | 59min | ~60min | ~1x ⚠️ |

### Step 3: 生成精炼版

将原始转录交给 AI 整理为精炼版。详细方法论见 [`精炼版整理指南.md`](精炼版整理指南.md)。

**核心原则：**

1. **保留论证过程**——不只提炼结论，保留因果链
2. **信息密度优先**——表格只为更高密度，不为做而做
3. **时间锚定**——每章标注原始音频时间区间 `【MM:SS-MM:SS】`
4. **完整性可验证**——精炼版必须覆盖原始转录 100% 内容

**精炼版 Prompt 模板：**

```
你是一个专业的音频转录精炼版整理专家。请将以下 ASR 原始转录整理为精炼版。

## 整理规则

1. **文头**：写明转录模型、时长、段数、完整性验证状态；以及录音来源、性质、关键人物
2. **章节**：按话题切换划分，每章标注时间区间【MM:SS-MM:SS】，中文数字编号
3. **内容**：保留论证过程和关键原话，不只提炼结论；数字必须精确；清理口语冗余
4. **表格**：对比、结构化数据、多维度并列时用表格；一句话能说清的不要硬做表格
5. **信息分层**：核心结论 → 支撑论据 → 具体数据 → 背景上下文 → 待定事项
6. **文末**：添加关键数据速览表（5-10个核心数据点）
7. **验证**：完成后核对原始转录段数与精炼版时间覆盖，确认无遗漏

现在请整理以下原始转录：

{粘贴原始转录内容}
```

## 📂 项目结构

```
Audio-Transcription-Skill/
├── README.md                       # 本文件
├── SKILL.md                        # WorkBuddy Skill 描述文件（本地安装时放在 ~/.workbuddy/skills/audio-transcription/）
├── requirements.txt                 # Python 依赖
├── .gitignore
├── LICENSE                         # MIT License
├── download_audio.py               # 音频下载脚本（B站/网盘/直链）
├── transcribe_qwen3_asr.py         # Qwen3-ASR 转录脚本
└── 精炼版整理指南.md                # 精炼版整理完整方法论 + Prompt 模板
```

## 🔧 技术细节

### 智能重叠去重原理

```
分片1: [0s ────── 30s]
分片2:           [25s ────── 55s]    ← 开头5s与分片1尾部重叠
分片3:                      [50s ────── 80s]

重叠区去重：分片2开头与分片1尾部相同的文本自动去除
→ 保证句子不断裂、不重复
```

### 断点续传

转录脚本自动保存进度到 `_checkpoint.json`，中断后重新运行相同命令即可恢复：

```bash
# 中断后恢复——输出路径一致，自动读取 checkpoint
python transcribe_qwen3_asr.py "output.mp3" -o "之前的输出.txt"
```

### 网盘下载原理

网盘分享页是 SPA 单页应用，需要 JavaScript 渲染。本工具使用 Playwright + 系统 Edge 浏览器：

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=False)
    # 自动导航到分享页 → 找到下载按钮 → 等待下载完成
```

> **关键**：`channel="msedge"` 调用系统 Edge 浏览器，无需额外下载 Chromium。
> 适用于极空间、百度网盘、阿里云盘等所有需要 JS 渲染的分享页。

### 完整性验证流程

转录和精炼完成后，自动执行验证：

1. **段数验证**：原始转录段数 vs 精炼版覆盖的时间区间
2. **时间覆盖**：精炼版最早时间 ≈ 00:00，最晚 ≈ 音频时长
3. **尾部检查**：原始转录最后 5-10 段内容在精炼版中有对应
4. **失败率**：转录失败率 >5% 需人工复查

## 🐛 故障排除

| 问题 | 解决方案 |
|------|---------|
| ffmpeg 未找到 | `winget install ffmpeg` |
| GPU 不可用 (Intel) | `pip install intel-extension-for-pytorch -i https://mirrors.aliyun.com/pypi/simple/` |
| GPU 不可用 (NVIDIA) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| GPU 不可用 (Apple MPS) | 确保 PyTorch ≥ 2.0，macOS ≥ 12.3，Apple Silicon Mac |
| ffmpeg 未找到 (macOS) | `brew install ffmpeg` |
| Playwright 网盘下载失败 | 确认有 Edge 浏览器，使用 `channel="msedge"` |
| 模型下载慢/失败 | 使用 ModelScope 镜像：`modelscope download Qwen/Qwen3-ASR-1.7B` |
| 转录中断 | 直接重新运行相同命令，自动从 checkpoint 恢复 |
| Chromium 安装 404 | 使用系统 Edge 代替，无需安装 Chromium |

## ⚠️ 重要规则

- **必须用 GPU 转录**，禁止纯 CPU 跑大模型推理任务（Intel Arc / NVIDIA CUDA / Apple MPS）
- **所有依赖和模型必须用国内镜像下载**，禁止翻墙
- Intel Arc GPU 用 IPEX (`intel-extension-for-pytorch`)，不是 CUDA
- Apple Silicon Mac 用 MPS 加速，确保 PyTorch ≥ 2.0
- pip 镜像：`https://mirrors.aliyun.com/pypi/simple/`
- 模型镜像：[ModelScope](https://modelscope.cn)，不用 HuggingFace

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Qwen3-ASR](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) - 阿里通义千问语音识别模型
- [Playwright](https://playwright.dev/python/) - 浏览器自动化
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - 视频下载工具

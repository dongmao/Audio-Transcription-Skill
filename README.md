# 🎙️ Audio Transcription Skill

> 音频转录精炼全流程——从下载到精炼，一键搞定

一键完成音频的**下载 → 语音转文字 → 精炼版整理**全流程。专为中文场景优化，支持 B站视频、网盘分享、HTTP 直链等多种来源。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🌐 **全来源覆盖** | B站视频、极空间/百度/阿里云盘分享、HTTP直链、本地文件 |
| ⚡ **SenseVoice 预览** | ~200M 小模型，Intel Arc 140V 实测 **~70x 实时率**，32秒完成39分钟音频 |
| 🎯 **Qwen3-ASR 精校** | 中文识别精度最高，1.7B 小模型 GPU 高效推理，~17-22x 实时率 |
| 📝 **精炼版整理** | 按主题重组 + 口语清理 + 信息分层 + 关键数据速览 |
| 🔗 **智能重叠去重** | 分片尾部 5s 重叠，避免句子断裂，自动拼接 |
| ♻️ **断点续传** | 长音频中断后可继续，不重复劳动 |
| ⚡ **全平台 GPU** | Intel XPU (Arc) / NVIDIA CUDA / Apple MPS，自动检测 |
| 🇨🇳 **国内镜像** | ModelScope 下载模型，阿里云 PyPI 镜像安装依赖 |
| ✅ **完整性验证** | 自动检查覆盖率、失败率，确保无遗漏 |

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 获取音频                                            │
│  B站 → yt-dlp | 网盘 → Playwright+Edge | 直链 → curl        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2.5a: SenseVoice 快速预览（可选，推荐）               │
│  ~200M 模型 | Intel Arc ~70x 实时率 | 32秒完成39分钟音频    │
│  → 获取全文章内容 → 精准调研背景知识                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Qwen3-ASR 精校转录（必须 GPU）                    │
│  智能分片(30s+5s重叠) → 批量推理 → 断点续传 → 重叠去重拼接  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: 生成精炼版（AI 整理）                              │
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

### Phase 1: 下载音频

```bash
# B站视频
python download_audio.py "https://www.bilibili.com/video/BV1xwQtBjEN1" -o "output.mp3"

# 极空间/百度网盘分享（自动启动 Edge 浏览器）
python download_audio.py "https://t3.znas.cn/xxxxx" -o "output.mp3"

# HTTP 直链
python download_audio.py "https://example.com/audio.mp3" -o "output.mp3"

# 本地文件（直接跳过下载步骤）
```

### Phase 2.5a: SenseVoice 快速预览（推荐）

**不要盲猜关键词，先听一遍内容。** 用 SenseVoice 小模型快速获取全文章内容，再精准调研背景知识。

| 属性 | 值 |
|------|------|
| 模型 | `iic/SenseVoiceSmall`（阿里 FunAudioLLM） |
| 参数量 | ~200M |
| 速度（Intel Arc 140V XPU） | **~70x 实时率**，39分钟音频约 32-35 秒 |
| 速度（CPU） | ~20x 实时率，39分钟约 2 分钟 |
| 精度 | 优于同等量级 Whisper，但不如 Qwen3-ASR-1.7B |

```bash
# 自动检测设备（XPU/CUDA/CPU）
python scripts/transcribe_sensevoice.py "录音.mp3" -o "录音_preview.txt"
```

> **注意**：`scripts/transcribe_sensevoice.py` 在 `scripts/` 子目录，不在 skill 根目录。

### Phase 2.5b: 预览文本驱动的精准调研

用预览文本提取陌生词汇（英文专有名词、疑似音译的人名/公司名），然后精准搜索验证，建立专有名词对照表：

```
ASR预览可能识别 → 正确名称
志源 → 智元机器人
物理智能 / 派 → Physical Intelligence (Pi)
hi world / agi world → AgiBot World
group one → Groot One
```

### Phase 3: Qwen3-ASR 精校转录

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

# 中断后断点续传（输出路径一致即可）
python transcribe_qwen3_asr.py "output.mp3" -o "之前的输出.txt"
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

| 阶段 | 设备 | 音频时长 | 耗时 | 实时率 |
|------|------|---------|------|--------|
| SenseVoice 预览 | Intel Arc 140V (16GB) | 39min | ~32秒 | **~70x** |
| Qwen3-ASR 精校 | Intel Arc 140V (16GB) | 42min | ~2分钟 | ~17-22x |
| Qwen3-ASR 精校 | NVIDIA RTX 4090 | 59min | ~3min | ~20x |
| Qwen3-ASR 精校 | Apple M1/M2/M3 (MPS) | 59min | ~8-12min | ~5-7x |
| Qwen3-ASR 精校 | CPU (i7) | 59min | ~60min | ~1x ⚠️ |

> ⚠️ **禁止纯 CPU 推理**。Intel Arc / NVIDIA CUDA / Apple MPS 任选其一。

### Phase 4: 生成精炼版

将原始转录交给 AI 整理为精炼版。详细方法论见 [`精炼版整理指南.md`](精炼版整理指南.md)。

**核心原则：**

1. **保留论证过程**——不只提炼结论，保留因果链
2. **信息密度优先**——表格只为更高密度，不为做而做
3. **时间锚定**——每章标注原始音频时间区间 `【MM:SS-MM:SS】`
4. **完整性可验证**——精炼版必须覆盖原始转录 100% 内容
5. **专有名词必须校正**——ASR 可能将英文名词音译为错误中文，需搜索验证

**精炼版 Prompt 模板：**

```
你是一个专业的音频转录精炼版整理专家。请将以下 ASR 原始转录整理为精炼版。

## 整理规则

1. **文头**：写明转录模型、时长、段数、完整性验证状态；录音来源、性质、关键人物
2. **章节**：按话题切换划分，每章标注时间区间【MM:SS-MM:SS】，中文数字编号
3. **内容**：保留论证过程和关键原话，不只提炼结论；数字必须精确；清理口语冗余
4. **表格**：对比、结构化数据、多维度并列时用表格；一句话能说清的不要硬做表格
5. **信息分层**：核心结论 → 支撑论据 → 具体数据 → 背景上下文 → 待定事项
6. **专有名词**：所有英文/专有名词需通过搜索确认正确中文名称，不确定时标注[待确认]
7. **ASR校正表**：文末附上 ASR 预览文本 → 正确名称的对照表（Phase 2.5b 产出）
8. **验证**：完成后核对原始转录段数与精炼版时间覆盖，确认无遗漏

现在请整理以下原始转录：

{粘贴原始转录内容}
```

## 📛 交付文件名规范

> ⭐ **禁止使用 B 站视频 ID（如 BVxxx）作为文件名**

### 标题格式

```
时间 + 节目类型 + 人物 + 内容主题
```

### 标题示例

```
2026年3月26日 · 小俊访谈 · 罗福莉（小米大模型负责人）——AI范式巨变与后训练新范式
```

### 正确 vs 错误文件名

| 类型 | 示例 |
|------|------|
| ✅ 正确 | `2026年5月13日_大摩直播_张雷×徐然×侯颖——周期论剑：AI超级周期与中国经济新格局_精炼版.md` |
| ❌ 错误 | `BV13G556aE2b_精炼版.md` |
| ❌ 错误 | `https___b23.tv_xxx_精炼版.md` |
| ❌ 错误 | `audio_20260515_精炼版.md` |

### 交付格式

| 项目 | 要求 |
|------|------|
| 格式 | Markdown (`.md`)，**不交付 .txt 版本** |
| 文件 | `{标题}_精炼版.md` + `{标题}_原始转录.md` |
| 编码 | UTF-8 |

## ✅ 快速检查清单

交付前必执行以下检查：

```bash
# 1. 原始转录段数
grep -c '^\[' 原始转录.md

# 2. 精炼版时间覆盖
grep -o '【[0-9:]*' 精炼版.md | sort -u

# 3. 检查文件命名（禁止 BV ID）
# ❌ BV13G556aE2b_精炼版.md
# ✅ 2026年5月13日_大摩直播_张雷×徐然×侯颖——周期论剑_精炼版.md
```

| 序号 | 检查项 | 说明 |
|------|--------|------|
| [ ] | 原始转录段数 | `grep -c '^\[' 原始转录.md` |
| [ ] | 精炼版时间覆盖区间 | 文头标注的覆盖区间 vs 音频总时长 |
| [ ] | 精炼版最后一章时间戳 | 应接近音频结尾（误差 <30秒） |
| [ ] | 专有名词已通过搜索确认并修正 | ASR 预览可能将英文名词音译错误 |
| [ ] | 不确定的专有名词标注了 [待确认] | — |
| [ ] | 文末附有 ASR 识别校正表 | 列出预览文本 → 正确名称的映射 |
| [ ] | 交付文件为 .md 格式 | **禁止**交付 .txt 版本 |
| [ ] | 标题格式符合规范 | **禁止**使用 B 站视频 ID |
| [ ] | 文头元信息完整 | 转录模型、时长、段数、转录耗时 |

## 📂 项目结构

```
Audio-Transcription-Skill/
├── README.md                       # 本文件
├── SKILL.md                        # WorkBuddy Skill 描述文件
├── requirements.txt                 # Python 依赖
├── .gitignore
├── LICENSE                         # MIT License
├── download_audio.py                # 音频下载脚本（B站/网盘/直链）
├── transcribe_qwen3_asr.py          # Qwen3-ASR 精校转录脚本
├── scripts/
│   └── transcribe_sensevoice.py    # SenseVoice 快速预览脚本（Phase 2.5a）
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

### 专有名词校对（血泪教训）

SenseVoice 预览会将英文名词音译为中文。典型错误案例：

| ASR预览识别 | 正确名称 | 备注 |
|------------|---------|------|
| "Happy Horse AI" | **快乐小马 (HappyHorse)** | 阿里巴巴 ATH 产品，2026年4月发布 |
| "Clean AI" | **可灵AI (Kling)** | 快手产品 |
| "C Dance" | **即梦 (Dreamina)** | 字节跳动产品 |
| "Jape" | **智谱GLM-5.1** | 智谱AI产品 |
| "Hi3" / "HY3" | **混元Hy3** | 腾讯大模型3.0代号 |
| 志源 | **智元机器人** | 智/志混淆 |
| 物理智能 / 派 | **Physical Intelligence (Pi)** | 意译/音译 |
| hi world / agi world | **AgiBot World** | 音节丢失 |

> **核心原则**：预览文本是信息来源，不是精确转录。所有英文/专有名词必须通过搜索确认。

### 完整性验证流程

转录和精炼完成后，自动执行验证：

1. **段数验证**：原始转录段数 vs 精炼版覆盖的时间区间
2. **时间覆盖**：精炼版最早时间 ≈ 00:00，最晚 ≈ 音频时长
3. **尾部检查**：原始转录最后 5-10 段内容在精炼版中有对应
4. **失败率**：转录失败率 >5% 需人工复查

## 🐛 故障排除

| 问题 | 解决方案 |
|------|---------|
| ffmpeg 未找到 | `winget install ffmpeg` (Windows) / `brew install ffmpeg` (macOS) |
| GPU 不可用 (Intel) | `pip install intel-extension-for-pytorch -i https://mirrors.aliyun.com/pypi/simple/` |
| GPU 不可用 (NVIDIA) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| GPU 不可用 (Apple MPS) | 确保 PyTorch ≥ 2.0，macOS ≥ 12.3，Apple Silicon Mac |
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
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 阿里 FunAudioLLM 快速预览模型
- [Playwright](https://playwright.dev/python/) - 浏览器自动化
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - 视频下载工具

---
name: audio-transcription
description: >
  音频转录全流程技能：下载→语音转文字→精炼版整理。
  支持B站视频、网盘分享、HTTP直链、本地文件的音频下载；
  使用 Qwen3-ASR-1.7B 模型 GPU 推理转录（Intel XPU / NVIDIA CUDA / Apple MPS）；
  AI 整理精炼版（话题分章、口语清理、信息分层、表格/列表）。
  全平台兼容：Windows/macOS/Linux，自动检测系统并适配。
  触发词：转录、语音转文字、ASR、音频转录、B站转录、录音整理、
  精炼版、Qwen3-ASR、音频下载、播客转录、会议录音整理
---

# 🎙️ Audio Transcription Skill

> GitHub: https://github.com/dongmao/Audio-Transcription-Skill

一键完成音频的**下载 → 语音转文字 → 精炼版整理**全流程。

## 核心特性

- **全来源覆盖**：B站视频、极空间/百度/阿里云盘分享、HTTP直链、本地文件
- **Qwen3-ASR 转录**：中文识别精度最高，1.7B 小模型 GPU 高效推理
- **SenseVoice 快速预览**：Phase 2.5a 小模型预览，32秒完成39分钟音频，用于指导精准调研
- **精炼版整理**：按主题重组 + 口语清理 + 信息分层 + 关键数据速览
- **智能重叠去重**：分片尾部5s重叠，避免句子断裂，自动拼接
- **断点续传**：长音频中断后可继续，不重复劳动
- **全平台 GPU**：Intel XPU (Arc) / NVIDIA CUDA / Apple MPS 自动检测
- **国内镜像**：ModelScope 下载模型，阿里云 PyPI 镜像
- **完整性验证**：自动检查覆盖率、失败率，确保无遗漏
- **跨平台智能适配**：自动检测操作系统，选择最佳工具和配置

## 工作流程

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1: 环境检测（一次性）                                       │
│  → 检测 GPU 类型 (Intel XPU / NVIDIA CUDA / Apple MPS / CPU)                 │
│  → 确认 qwen-asr、funasr、modelscope、librosa、playwright 已安装          │
│  → 未安装则用国内镜像 pip install                                 │
│                                                                  │
│  Phase 2: 获取音频                                                │
│  → B站链接：yt-dlp 下载音频                                      │
│  → 网盘分享：Playwright + 系统Edge 浏览器自动下载                │
│  → HTTP直链：curl 下载                                           │
│  → 本地文件：直接使用                                             │
│  → 记录音频时长用于完整性验证                                     │
│                                                                  │
│  ⭐ Phase 2.5a: SenseVoice 快速预览（2026-05-15新增）         │
│  → 用 SenseVoice-Small (~200M) 快速跑一遍音频                    │
│  → 32秒完成39分钟音频（~70x实时率），获取全文章内容               │
│  → 清除特殊标签（<|zh|><|NEUTRAL|><|Speech|>等元数据）         │
│  → 预览文本用于 Phase 2.5b 精准调研                              │
│                                                                  │
│  ⭐ Phase 2.5b: 预览文本驱动的精准调研（2026-05-15新增）       │
│  → 从 SenseVoice 预览文本中提取陌生词汇/专有名词                  │
│  → 针对这些词搜索验证（不能盲猜关键词）                            │
│  → 建立专有名词对照表（ASR可能识别 → 正确名称）                  │
│  → 此步骤直接影响精炼版准确率，必须认真执行                        │
│                                                                  │
│  Phase 3: 转录（Qwen3-ASR，必须 GPU）                             │
│  → 智能分片：30s片长 + 5s重叠                                    │
│  → GPU加速批量推理（自动调批次大小）                              │
│  → dtype 降级链：bfloat16 → float16 → float32（自动）           │
│  → 断点续传：checkpoint.json 记录进度                            │
│  → 重叠去重：自动拼接，避免句子断裂                              │
│  → 失败降级：批次失败→逐条重试（已内置）                         │
│                                                                  │
│  Phase 4: 整理精炼版（AI 核心）                                   │
│  → 按主题重组章节，标注时间锚定                                   │
│  → 对照专有名词表修正ASR识别错误                                  │
│  → 不确定的标注 [待确认]，末尾附校正表                           │
│  → 去除口语冗余，保留论证过程                                     │
│  → 结构化数据用表格，信息分层组织                                 │
│  → 文头元信息 + 文末关键数据速览                                 │
│  → 完整性验证：段数 + 时间覆盖双重校验                           │
│                                                                  │
│  Phase 5: 交付                                                    │
│  → 精炼版 .md + 原始转录 .md（两个文件）                         │
│  → 完整性验证通过后交付                                           │
└──────────────────────────────────────────────────────────────────┘
```

## Phase 1: 环境检测

### 首次使用前安装依赖

```bash
# 核心依赖（阿里云镜像）
pip install qwen-asr funasr modelscope librosa soundfile yt-dlp playwright requests -i https://mirrors.aliyun.com/pypi/simple/

# Playwright 浏览器（网盘下载必需）
# macOS 不需要额外安装，使用系统浏览器

# Intel XPU 加速（Arc GPU 用户）
pip install intel-extension-for-pytorch -i https://mirrors.aliyun.com/pypi/simple/

# NVIDIA CUDA 加速
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ffmpeg（音频处理必需，macOS）
brew install ffmpeg
```

### 环境验证

```bash
# 检查 GPU（跨平台）
python -c "
import torch, platform
print(f'平台: {platform.system()} {platform.machine()}')
print('XPU:', hasattr(torch,'xpu') and torch.xpu.is_available())
print('CUDA:', torch.cuda.is_available())
if platform.system() == 'Darwin' and platform.machine() in ['arm64', 'aarch64']:
    print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
"

# 检查 qwen-asr
python -c "from qwen_asr import Qwen3ASRModel; print('qwen-asr OK')"

# 检查 funasr（SenseVoice 预览用）
python -c "import funasr; print('funasr OK:', funasr.__version__)"

# 检查 Playwright
python -c "from playwright.sync_api import sync_playwright; print('playwright OK')"
```

> **重要规则**：
> - **必须用 GPU/XPU/MPS 转录，禁止纯 CPU 跑大模型任务**
> - 所有依赖和模型必须用国内镜像下载，禁止翻墙
> - Intel Arc GPU 需安装 IPEX (intel-extension-for-pytorch)，不是 CUDA
> - Apple Silicon Mac 使用 MPS 加速，无需额外安装
> - pip 镜像：`https://mirrors.aliyun.com/pypi/simple/` 或 `https://pypi.tuna.tsinghua.edu.cn/simple`
> - 模型镜像：ModelScope（modelscope.cn），不用 HuggingFace
> - 脚本自动检测平台和 GPU 类型，无需手动配置

## Phase 2: 获取音频

### 2.1 统一下载脚本（推荐）

```bash
# 自动识别来源类型
python download_audio.py "<URL>" -o "录音.mp3"

# 支持所有来源：B站、网盘、直链
# 自动检测来源类型，选择最优下载方式
```

### 2.2 手动下载

**B站视频/直播：**
```bash
python3 -m yt_dlp -x --audio-format mp3 --audio-quality 0 "<URL>" -o "录音.mp3"
```

**网盘分享（极空间/百度/阿里云盘等）：**

网盘分享页面是 SPA 单页应用，需要浏览器渲染。使用 Playwright + 系统浏览器自动下载（跨平台）：

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # 脚本自动选择最佳浏览器：Windows→Edge, macOS→Chrome, Linux→Chromium
    browser = p.chromium.launch(channel="auto", headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(网盘分享URL)
    page.wait_for_load_state("networkidle")
    # 找到文件项 → 点击下载 → 等待下载完成 → 保存
    browser.close()
```

**HTTP 直链：**
```bash
curl -L -o "录音.mp3" "<URL>"
```

### 2.3 下载后验证

```bash
ffprobe -i "录音.mp3" -show_entries format=duration,size -v quiet -of csv="p=0"
```

> **重要**：记录实际音频时长，用于后续完整性验证。

## ⭐ Phase 2.5a: SenseVoice 快速预览（必须执行！2026-05-15新增）

> **核心思想**：不要盲猜关键词，而是先"听"一遍内容，再精准调研。

**为什么需要预览？**

| 旧流程问题 | 新流程优势 |
|-----------|-----------|
| 根据视频标题猜关键词 | 从实际转录文本提取词汇 |
| "智元"查不到 → 写错"智源" | 从预览文本找到"志源" → 搜索确认"智元" |
| "Sharp"搜不到 | 从预览文本找到类似发音 → 搜索确认 |
| 背景调研依赖预训练知识 | 直接看转录文本，100% 准确 |

**SenseVoice-Small 预览模型**：

| 属性 | 值 |
|------|------|
| 模型 | `iic/SenseVoiceSmall`（阿里 FunAudioLLM） |
| 参数量 | ~200M |
| 速度（Intel Arc 140V XPU） | **~70x 实时率**，39分钟音频约 32-35 秒 |
| 速度（CPU） | ~20x 实时率，39分钟约 2 分钟（仍可接受） |
| 精度 | 优于同等量级 Whisper，但不如 Qwen3-ASR-1.7B |
| 输出标签 | `<|zh|><|NEUTRAL|><|Speech|>` 等（需清除） |

**运行预览脚本**：

```bash
# 自动检测设备（XPU/CUDA/CPU）
python scripts/transcribe_sensevoice.py "录音.mp3" -o "录音_preview.txt"

# 指定设备
python scripts/transcribe_sensevoice.py "录音.mp3" -d "xpu" -o "录音_preview.txt"
```

**选项**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出文件路径 | `{原文件名}_SenseVoice_preview.txt` |
| `-s, --segment-len` | 分片长度（秒） | 30 |
| `--overlap` | 重叠长度（秒） | 3 |
| `-d, --device` | 推理设备 | 自动检测 |
| `--batch-size` | 批次大小 | 8 |

**预览脚本会自动清除特殊标签**（`<|zh|>`、`<|NEUTRAL|>`、`<|Speech|>` 等元数据标签），输出纯净文本。

**性能参考**：

| 设备 | 音频时长 | 预览耗时 | 实时率 |
|------|---------|---------|--------|
| Intel Arc 140V (16GB) | 39min | ~32秒 | ~70x |
| CPU (i7) | 39min | ~2分钟 | ~20x |
| NVIDIA RTX 4090 | 39min | ~25秒 | ~90x |

## ⭐ Phase 2.5b: 预览文本驱动的精准调研（必须执行！2026-05-15新增）

> **核心思想**：从 SenseVoice 预览文本中提取词汇，而不是盲猜。

### 步骤一：从预览文本提取陌生词汇

读取 `录音_preview.txt`，扫描以下模式：

```
1. 英文专有名词（连续大写字母）："Pi", "Figure", "Sharp", "RT-1"
2. 疑似音译的人名/公司名（常见错误模式）：
   - "志源" → 可能是"智元"（智/志混淆）
   - "物理智能" → 可能是"Physical Intelligence"
   - "hi world" → 可能是"AgiBot World"
   - "group one" → 可能是"Groot One"
3. 不确定的技术术语
```

### 步骤二：精准搜索验证

对于每个提取的陌生词汇：

```bash
# 示例：搜索"志源机器人"确认是否"智元机器人"
WebSearch: "志源 机器人 具身智能"

# 示例：搜索"hi world 具身智能"确认是否"AgiBot World"
WebSearch: "AgiBot World 具身智能 数据集"

# 示例：搜索"Sharp 机器人 乒乓球"
WebSearch: "Sharp Robotics 机器人 乒乓球 CES"
```

### 步骤三：建立专有名词对照表

```
ASR预览可能识别    →  正确名称（搜索确认）
志源               →  智元机器人（人形机器人公司）
物理智能 / Pi / 派  →  Physical Intelligence (Pi)
hi world / AGIBOT  →  AgiBot World（智元开源数据集）
group one          →  Groot One（英伟达具身基础模型）
sharp              →  Sharp Robotics
figure             →  Figure AI
```

**注意**：搜索时要找**权威来源**（官网、新闻、财经媒体），不要凭模型记忆猜测。

### 步骤四：预览文本对背景调研的价值

即使 SenseVoice 预览有音译错误，它仍能告诉你：

- ✅ 录音的话题方向（具身智能/机器人数据）
- ✅ 提到的公司类型（人形机器人、数据采集）
- ✅ 技术路线（仿真、遥操、动捕、视频）
- ✅ 大致的时间结构和内容分布
- ⚠️ 具体人名/公司名需要搜索验证

---

## Phase 3: 转录

### 运行转录脚本

```bash
python transcribe_qwen3_asr.py "录音.mp3" [options]
```

**选项：**

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

### 运行示例

```bash
# 基本用法（自动检测GPU）
python transcribe_qwen3_asr.py "录音.mp3"

# 指定 Intel XPU
python transcribe_qwen3_asr.py "录音.mp3" -d "xpu:0"

# 指定 Apple MPS
python transcribe_qwen3_asr.py "录音.mp3" -d "mps"

# 恢复中断的转录（自动读取 checkpoint）
python transcribe_qwen3_asr.py "录音.mp3" -o "之前的输出.txt"
```

### 输出格式

```
[00:00] 朋友大家上午好欢迎来到一周一度的大摩宏观策略堂...
[00:25] 我们的首席策略师罗拉...
[00:50] 现在的局势依然是比较脆弱的...
```

### 性能参考

| 设备 | 音频时长 | 转录耗时 | 实时率 |
|------|---------|---------|--------|
| Intel Arc 140V (16GB) | 59min | 3.1min | ~19x |
| NVIDIA RTX 4090 | 59min | ~3min | ~20x |
| Apple M1/M2/M3 (MPS) | 59min | ~8-12min | ~5-7x |
| CPU (i7) | 59min | ~60min | ~1x ⚠️ |

## Phase 4: 整理精炼版

> **这是本 skill 最核心的部分**。详细指南见 `references/精炼版整理指南.md`。
>
> ⚠️ **重要**：整理前必须先执行 Phase 2.5 背景调研，建立专有名词表。整理时对照修正ASR识别错误，不确定的标注[待确认]，末尾附ASR识别校正表。

### 4.1 核心原则

1. **保留论证过程，不止提炼结论**——用户需要掌握详细信息
2. **信息密度优先**——表格为了更高信息密度，不为做而做
3. **时间锚定**——每个章节标注原始音频时间区间
4. **完整性可验证**——精炼版必须覆盖原始转录 100% 内容

### 4.2 文头元信息（必须）

```markdown
> *转录模型：XXX | 转录时间：YYYY-MM-DD | 设备：XXX*
> *音频时长：XX分XX秒 | 转录段数：XX段 | 完整性验证：✅ 通过（XX/XX段，X%失败，覆盖00:00-XX:XX）*

> *录音来源：XXX | 文件：XXX | 大小：XXX*
> *录音性质：XXX | 参会方/嘉宾：XXX*
```

### 4.3 章节划分

- **按话题切换划分**，不按等时间切
- 章节标题后标注时间区间：`## 章节名 【MM:SS-MM:SS】`
- 章节编号用中文数字（一、二、三...）
- 30分钟录音通常 8-15 个章节
- 子议题用 `###` + 编号（1.1, 1.2...）

### 4.4 内容整理手法

**何时用表格**：
- ✅ 多维度对比、结构化数据、时间线/里程碑、多方观点并列
- ❌ 一句话能说清、叙述性内容、单一数值、情感/语气描述

**保留 vs 改写**：
- 关键定义/论断/决策 → 保留原话
- 数字/金额/时间节点 → 精确保留
- 逻辑推导过程 → 保留因果链
- 背景叙述 → 可改写精炼

**口语清理**：
- 重复词保留一次，口头禅删除
- 自我纠正只保留最终结论
- 模糊表达原样保留，不擅自精确化
- ASR 同音字错误根据上下文修正

**信息分层**（每个章节）：
1. 核心结论/决策
2. 支撑论据/过程
3. 具体数据
4. 背景/上下文
5. 待定/未决事项

### 4.5 特殊内容类型

| 类型 | 处理要点 |
|------|---------|}
| 会议录音 | 标注参会方角色；区分"已决定"vs"待讨论"vs"有分歧" |
| 访谈/播客 | 区分主持人和嘉宾；嘉宾核心观点加粗突出 |
| 技术讲解 | 专有名词保留原文+中文注释；不确定处标 `[待确认]` |

### 4.6 文末速览表（推荐）

```markdown
## 关键数据速览

| 指标 | 数值 |
|------|------|
| XXX | XXX |
```

只放最关键的 5-10 个数据点，让读者 10 秒抓住要点。

### 4.7 精炼版 Prompt 模板（2026-06-13 更新：加入深度要求）

整理精炼版时，AI 助手应遵循以下 prompt 逻辑：

```
你是一个专业的音频转录精炼版整理专家。请将以下 ASR 原始转录整理为精炼版。

## 整理规则

1. **文头**：写明转录模型、转录时间、设备、音频时长、转录段数、完整性验证状态；
   以及录音来源、性质、关键人物
2. **标题格式**：时间 + 节目类型 + 人物 + 内容主题
   示例：2026年3月26日 · 小俊访谈 · 罗福莉（小米大模型负责人）——AI范式巨变与后训练新范式
3. **章节**：按话题切换划分，每章标注时间区间【MM:SS-MM:SS】，中文数字编号
4. **内容**：保留论证过程和关键原话，不只提炼结论；数字必须精确；清理口语冗余
5. **深度要求**：确保每个关键概念在文中被解释"为什么重要"；每个重大决定/转型有完整的"前因→后果"链条；保留对话张力（谁问了什么、为什么这么问）；读者没听原录音也能完全搞清楚在聊什么
6. **表格**：对比、结构化数据、多维度并列时用表格；一句话能说清的不要硬做表格
7. **信息分层**：核心结论 → 支撑论据 → 具体数据 → 背景上下文 → 待定事项
8. **文末**：添加关键数据速览表（5-10个核心数据点）
8. **⚠️ 验证（完成后必须执行）**：
   - 统计原始转录段落数（grep -c '^\[' 原始转录.md）
   - 统计精炼版覆盖的时间戳数量（grep -o '【[0-9:]*' 精炼版.md | sort -u）
   - 对比原始转录时间段数 vs 精炼版覆盖时间范围
   - 检查精炼版最后一个章节的时间戳是否接近音频结尾
   - 如发现有遗漏，补充遗漏内容后重新验证

现在请整理以下原始转录：

{原始转录内容}
```

## Phase 5: 完整性验证（必须步骤）

> ⚠️ **血泪教训**：2026年4月30日，3小时41分B站访谈转录，因未做转录文件时间线校验，精炼版只覆盖到1:40，遗漏后半段2小时内容。此后必须严格执行以下验证流程。

**完整验证流程分为两个阶段：**

### 阶段A：转录完成后的校验（进入精炼前）

转录完成后、开始整理前，必须执行：

```bash
python3 -c "
import re
content = open('原始转录.txt', 'r', encoding='utf-8').read()
lines = [l for l in content.strip().split('\n') if l.strip()]
times = re.findall(r'\[(\d+:\d{2}(?::\d{2})?)\]', content)
failed = sum(1 for l in lines if '转录失败' in l)
print(f'原始转录: {len(lines)}段')
print(f'时间范围: {times[0] if times else \"N/A\"} - {times[-1] if times else \"N/A\"}')
print(f'失败段数: {failed} ({100*failed/len(lines):.1f}%)' if lines else 'N/A')
"
```

**校验要点：**
1. 最早时间戳 ≈ 00:00
2. 最晚时间戳 ≈ 音频时长（3小时音频应该是 3:xx:xx）
3. 失败率 < 5%

⚠️ **如果最晚时间戳远小于音频时长，说明转录不完整，必须重新转录或补全。**

### 阶段B：精炼版完成后的校验（交付前）

**必须执行的验证清单：**

1. **统计原始转录段落数**：计算原始转录文件的行数/段落数
2. **确认音频实际时长**：通过 ffprobe 获取精确时长
3. **检查精炼版时间覆盖**：
   - 最早时间戳 ≈ 00:00
   - 最晚时间戳 ≈ 音频结尾
   - 时间戳之间无明显大段跳跃
4. **检查转录失败率**：失败率 >5% 需人工复查
5. **尾部检查**：原始转录最后 5-10 段内容在精炼版中有对应

### 验证命令

```bash
python3 -c "
import re
content = open('原始转录.txt', 'r', encoding='utf-8').read()
lines = content.strip().split('\n')
times = re.findall(r'\[(\d+:\d{2}(?::\d{2})?)\]', content)
failed = sum(1 for l in lines if '转录失败' in l)
print(f'原始转录: {len(lines)}段, 时间范围 {times[0]}-{times[-1]}, 失败{failed}段')

refined = open('精炼版.md', 'r', encoding='utf-8').read()
refined_times = re.findall(r'【(\d+:\d{2}(?::\d{2})?)', refined)
print(f'精炼版: {len(refined_times)}个时间区间, {refined_times[0]}-{refined_times[-1]}')
"
```

### ⚠️ 遗漏根因分析与防范

**为什么会遗漏？**
- 直接按上下文顺序整理，没有先检查转录文件的时间范围
- 长音频转录结果可能按时间顺序排列，也可能按其他顺序
- 如果在某个位置截断后继续整理，无法通过上下文发现遗漏

**正确做法（系统性章节划分）：**
1. 转录完成后，读取转录文件首行和末行的时间戳
2. 确认完整覆盖整个音频时间范围
3. 将转录内容按时间顺序系统性划分章节
4. 每个章节覆盖特定时间段，确保无遗漏

```python
# 正确的章节划分流程
import re

# 1. 读取转录文件
with open('原始转录.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. 提取所有时间戳
times = []
for line in lines:
    match = re.match(r'\[(\d+:\d{2}(?::\d{2})?)\]', line)
    if match:
        times.append(match.group(1))

# 3. 确认首尾时间戳
print(f"首: {times[0]}, 尾: {times[-1]}")
# 如果尾时间 << 音频时长，说明转录不完整

# 4. 按时间顺序系统性划分章节（而非按内容顺序）
# 将整个时间范围分成N段，每段对应一个章节
```

### 验证通过标准

- 原始转录失败段数 ≈ 0（允许 <5%）
- 精炼版最早时间 ≈ 00:00
- 精炼版最晚时间 ≈ 音频时长（误差 <30秒）
- 精炼版章节数 × 平均段落数 ≈ 原始转录总段数（允许 ±5%）
- 无明显时间跳跃（相邻章节时间差 < 章节时长 + 60秒）
- 尾部检查通过：原始转录最后 5-10 段内容在精炼版中有对应

### ⚠️ 验证命令速查（交付前必须执行）

```bash
# 1. 原始转录段落数
grep -c '^\[' 原始转录.md

# 2. 精炼版所有时间戳（去重排序）
grep -o '【[0-9:]*' 精炼版.md | sort -u

# 3. 精炼版时间覆盖区间
grep -o '【[0-9:]*' 精炼版.md | head -1
grep -o '】[0-9:]*' 精炼版.md | tail -1

# 4. 音频实际时长
ffprobe -i 音频.mp3 -show_entries format=duration -v quiet -of csv="p=0"

# 5. 验证通过后再更新文头完整性状态为 ✅ 通过
```

验证通过后，更新精炼版文头的完整性验证状态为 ✅ 通过。

## 交付规范

| 项目 | 要求 |
|------|------|
| 格式 | Markdown (.md)，**不交付 .txt 版本** |
| 文件 | `{标题}_精炼版.md` + `{标题}_原始转录.md` |
| 标题生成 | **禁止用 B 站视频 ID（如 BVxxx）作为文件名**。应从音频内容提取：时间（YYYY年MM月DD日）+ 节目类型（大摩直播/播客/访谈/电话会议等）+ 人物（嘉宾姓名+机构/职位）+ 内容主题（核心议题） |
| 标题格式 | `时间 + 节目类型 + 人物 + 内容主题` |
| 标题示例 | `2026年3月26日 · 小俊访谈 · 罗福莉（小米大模型负责人）——AI范式巨变与后训练新范式` |
| 原始转录 | 将 .txt 复制为 .md，不做修改 |
| 编码 | UTF-8 |
| 交付方式 | 使用 deliver_attachments 工具同时交付两个文件 |

## 故障排除

| 问题 | 解决方案 |
|------|---------|}
| ffmpeg 未找到 | macOS: `brew install ffmpeg` |
| GPU 不可用 (Intel) | `pip install intel-extension-for-pytorch -i https://mirrors.aliyun.com/pypi/simple/` |
| GPU 不可用 (NVIDIA) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| GPU 不可用 (Apple MPS) | 确保 PyTorch ≥ 2.0，macOS ≥ 12.3，Apple Silicon Mac |
| Playwright 网盘下载失败 | 脚本自动选择系统浏览器，无需手动配置 |
| 模型下载慢/失败 | 使用 ModelScope 镜像：`modelscope download Qwen/Qwen3-ASR-1.7B` |
| 转录中断 | 直接重新运行相同命令，自动从 checkpoint 恢复 |
| Chromium 安装 404 | 使用系统 Edge 代替，无需安装 Chromium |

## 脚本说明

| 脚本 | 用途 |
|------|------|}
| `download_audio.py` | 音频下载（B站/网盘/直链，自动识别来源） |
| `scripts/transcribe_sensevoice.py` | SenseVoice-Small 快速预览（Phase 2.5a，用于指导精准调研） |
| `transcribe_qwen3_asr.py` | Qwen3-ASR 精校转录（Phase 3，高精度） |

## 参考文档

| 文档 | 用途 |
|------|------|}
| `references/精炼版整理指南.md` | 精炼版整理的完整方法论和示例 Prompt |

## 实践经验与教训

> 来自 2026 年 3-5 月、累计 20+ 个转录任务的实战总结。

### 教训一：完整性验证是最大坑（⚠️ 高频踩坑）

- **现象**：精炼版整理完后，发现只覆盖了原始转录的前半段，后半段完全遗漏
- **根因**：按上下文顺序逐段整理，没有先系统性检查原始转录的时间覆盖范围
- **正确做法**：
  1. 转录完成后立即检查首尾时间戳，确认覆盖完整音频时长
  2. 精炼版整理完后，用 grep 统计段落数 + 时间覆盖，对比原始转录
  3. 最终交付前，人工检查精炼版最后一章的时间戳是否接近音频结尾
- **验证命令（交付前必执行）**：
  ```bash
  # 原始转录段数
  grep -c '^\[' 原始转录.md
  # 精炼版时间戳列表（应覆盖原始转录主要时间段）
  grep -o '【[0-9:]*' 精炼版.md | sort -u
  ```

### 教训二：章节时间标签容易出错

- **现象**：修复重复/错误章节后，时间标签没有同步修正，导致时间戳混乱
- **正确做法**：
  - 修改章节内容时，必须同步核对并修正章节标题中的时间区间【MM:SS-MM:SS】
  - 用 `grep '^##' 精炼版.md` 列出所有章节标题，逐一核对时间标签
  - 删除重复章节时，确保只删除旧版本，保留修正后的版本

### 教训三：长音频转录完整性

- **现象**：1小时以上的音频，转录结果可能不完整（尾部长尾丢失）
- **正确做法**：
  - 转录脚本应设置最大分段数上限，避免内存溢出导致截断
  - 转录完成后立即用 ffprobe 获取音频时长，与转录结果最晚时间戳对比
  - 时间差 >60秒 视为不完整，需排查原因（内存、crash、分片逻辑错误）

### 教训四：GPU 必须强制使用

- **现象**：脚本在 CPU 模式下也能跑，但速度是 GPU 的 1/10，导致超时或用户取消
- **正确做法**：
  - 转录脚本开头检测 GPU 可用性，CPU 模式下打印警告但不强制退出（给用户选择权）
  - 在 skill 文档和 prompt 中明确标注：建议使用 GPU，CPU 速度约 1x 实时率
  - Intel XPU 用户需确认 `intel-extension-for-pytorch` 已安装

### 教训五：专有名词校对（⚠️ 血泪教训，2026年5月12日更新）

- **现象**：ASR 对人名、公司名、产品名、技术术语识别错误，精炼版延续错误，导致大量错别字
- **正确做法**：转录前必须执行背景调研，搜索确认专有名词
- **典型错误案例**（2026年5月12日 杰富瑞闭门会转录，已通过搜索验证）：

| ASR识别 | 正确名称 | 备注 |
|---------|----------|------|
| "Happy Horse AI" | **快乐小马 (HappyHorse)** | 阿里巴巴ATH产品，2026年4月发布 |
| "Clean AI" | **可灵AI (Kling)** | 快手产品 |
| "C Dance" | **即梦 (Dreamina)** | 字节跳动产品 |
| "Jape" | **智谱GLM-5.1** | 智谱AI产品（上下文提到"GLM 5.1"） |
| "JPod" | **阶跃星辰** | 上下文提到"Step系列" |
| "GAIPU / Zepu" | **腾讯混元团队** | 技术分享部分，来源为腾讯 |
| "Hi3" / "HY3" | **混元Hy3** | 腾讯大模型3.0英文代号（Hunyuan Hy3） |
| 数字/英文混合 | K2.6、K two point six | 统一为 **Kimi K2.6** |

#### ⭐ 转录前必须执行的背景调研（强制步骤）

**在下载音频后、转录开始前，必须先搜索相关领域的背景信息**：

1. **搜索关键词**：
   - 视频/音频的主题（如"杰富瑞 中国互联网 AI"）
   - 视频中提到的公司名（如"快乐小马"、"可灵AI"、"即梦"、"混元Hy3"）
   - 视频中提到的人名（搜索确认正确拼写）
   - 技术术语和产品名称

2. **建立专有名词对照表**：
   ```
   ASR可能识别为 → 正确名称
   Happy Horse AI → 快乐小马 (HappyHorse)
   Clean AI → 可灵AI (Kling)
   C Dance → 即梦 (Dreamina)
   Jape → 智谱GLM-5.1
   GAIPU/Zepu → 腾讯混元团队
   Hi3 → 混元Hy3
   ```

3. **搜索工具**：
   - 用 `WebSearch` 搜索权威来源（官网、新闻、财经媒体）
   - 不要凭模型记忆猜测，要找实际证据
   - 特别注意：很多"查无此名"的实际上可能是真实产品（如HappyHorse）

4. **精炼版整理时**：
   - 对照专有名词表修正明显的 ASR 同音字错误
   - 遇到不确定的专有名词，**必须标注 `[待确认]`** 而不是猜测
   - 在精炼版末尾附上 **ASR识别校正表**，列出所有被修正的专有名词

#### 常见ASR错误类型

| 错误类型 | 示例 | 修正 |
|----------|------|------|
| 英文音译错误 | "Happy Horse" | 快乐小马 (HappyHorse) |
| 品牌名混淆 | "Clean AI" | 可灵AI (Kling) |
| 拼音首字母 | "C Dance" | 即梦 (Dreamina) |
| 产品代号 | "Hi3" | 混元Hy3 |
| 数字+英文混合 | "K two point six" | K2.6 |

### 教训六：精炼版不是摘要，是再整理（⚠️ 高频踩坑，2026-06-13 升级版）

- **现象**：精炼版写成摘要形式，丢失了大量细节、论证过程、对话上下文，用户反馈"太简单了看不懂他们在聊什么"
- **根因**：AI 把精炼理解成了"压缩和提炼"，但实际上用户需要的是"**让没听原录音的人能完全搞清楚在聊什么**"
- **三次迭代验证**（2026-06-13，张小珺访谈阳萌 3h38m）：
  - V1（16KB）：浓缩版，信息密度高 → 用户反馈"太简单了看不懂"
  - V2（37KB）：补全对话上下文、推理链、概念解释、对话张力 → 用户认可
- **正确做法**：
  - 精炼版的目标是**让读者能读懂原文在聊什么**，不是**缩短长度**
  - 保留完整推理链：不仅说"做了什么"，更要说明"**为什么这么做**"的推导过程
  - 保留对话张力：谁问了什么、为什么这么问、这个问题戳中了什么矛盾
  - **解释每个概念为什么重要**：一个框架/比喻/概念出现时，要说明它在整段对话中扮演什么角色
  - 保留关键原话和场景细节：让读者感受到对话的氛围和人物的个性
  - 用表格/列表提高结构化数据的可读性，但核心叙述必须完整
- **精炼版的检查标准（新）**：
  - ✅ 读者没听过原录音，看完能清楚这个人在聊什么、为什么聊这些、他的核心观点是什么
  - ✅ 每个关键概念在文中至少被解释一次"为什么重要"
  - ✅ 每个重大决定/转型都有完整的"前因→后果"链条
  - ❌ 是否看起来像"太长不想读"（信息密度高 ≠ 字数少）

### 教训七：交付文件规范

- **现象**：交付了 .txt 版本，或者标题格式不统一，用户需要手动重命名
- **正确做法**：
  - 只交付 .md 文件（不交付 .txt），原始转录 .txt 复制为 .md
  - 标题格式统一为：`时间 + 节目类型 + 人物 + 内容主题`
  - 使用 `deliver_attachments` 工具一次性交付两个文件（精炼版 + 原始转录）
  - 文头元信息完整：转录模型、转录时间、设备、音频时长、段数、完整性状态

### 教训八：背景调研不能盲猜，必须用 SenseVoice 预览驱动（⭐ 2026-05-15 新增）

- **现象**：根据视频标题/元信息盲猜关键词去搜索，导致"智元"写成"智源"、"Sharp"搜不到
- **根因**：在不知道录音具体内容的情况下，只能靠预训练知识猜测；很多专有名词不在预训练数据中
- **正确做法**：Phase 2.5a 先用 SenseVoice-Small 跑一遍预览（32秒完成39分钟），Phase 2.5b 从预览文本中提取陌生词汇，再精准搜索验证

**实战数据**（2026-05-15，BV1uQ5Y6mExN，39分钟）：

| 指标 | 数值 |
|------|------|
| SenseVoice 预览耗时 | **32秒**（XPU，70x 实时率） |
| Qwen3-ASR 精校耗时 | **112秒**（20.8x 实时率） |
| 预览速度 vs 精校 | 快 **3.5倍** |
| 关键词识别（智元） | 预览文本有"志源" → 搜索确认"智元" |
| 关键词识别（Physical Intelligence） | 预览文本有"物理智能"/"派" → 搜索确认 |

**SenseVoice 预览的常见音译偏差模式**：

| 预览文本（ASR） | 正确名称 | 偏差类型 |
|---------------|---------|---------|
| 志源 | 智元机器人 | 智/志混淆 |
| 物理智能 / 派 | Physical Intelligence (Pi) | 意译/音译 |
| hi world / agi world | AgiBot World | 音节丢失 |
| group one | Groot One | 语义相近 |
| sharp | Sharp Robotics | 首字母大写丢失 |
| 志得 / 吉得 | GID（通用具身智能数据平台）[待确认] | 需二次验证 |

> **核心原则**：预览文本是**信息来源**，不是精确转录。可以用它识别话题方向和提取关键词，但具体人名/公司名必须通过搜索确认。

### 快速检查清单（每次交付前）

- [ ] 原始转录段数 = grep -c '^\[' 原始转录.md
- [ ] 精炼版时间覆盖 = 文头标注的覆盖区间
- [ ] 精炼版最后一章时间戳 ≈ 音频结尾（误差 <30秒）
- [ ] 所有章节标题的时间标签已核对（grep '^##' 精炼版.md）
- [ ] **⭐ 专有名词已通过搜索确认并修正**（不能用模型记忆，必须搜索验证）
- [ ] **⭐ 不确定的专有名词标注了 [待确认]**
- [ ] **⭐ 精炼版末尾附有 ASR识别校正表**
- [ ] 交付文件为 .md 格式（非 .txt）
- [ ] 标题格式符合规范（禁止使用 B 站视频 ID 作为文件名）
- [ ] 文头元信息完整
- [ ] **⭐ 精炼版深度自检：没听原录音的人能完全搞懂吗？**

### 教训九：文件名必须用标题而非视频 ID（⭐ 2026-05-15 新增）

- **现象**：转录完成后用 B 站视频 ID（如 `BV13G556aE2b`）作为文件名，导致文件名没有信息量，用户难以识别内容
- **根因**：skill 交付规范只说了标题格式，没说清楚"标题"从哪里来，容易直接用 URL 或文件名中的 ID
- **正确做法**：
  1. 转录完成后，从音频内容（精炼版章节标题、文头元信息、参会人物）中提取四个要素
  2. 拼装为规范标题：`YYYY年MM月DD日 + 节目类型 + 人物 + 内容主题`
  3. 文件名 = `规范标题_精炼版.md` 和 `规范标题_原始转录.md`
  4. 绝对禁止用 B 站视频 ID、URL 片段、或时间戳作为文件名主干
- **文件名命名检查**（交付前必须执行）：
  ```bash
  # ❌ 错误：使用了视频ID
  BV13G556aE2b_精炼版.md
  https___b23.tv_xxx_精炼版.md
  audio_20260515_精炼版.md

  # ✅ 正确：使用规范标题
  2026年5月13日_大摩直播_张雷（基础材料）×徐然（Richard_中银金融首席）×侯颖——周期论剑：AI超级周期与中国经济新格局_精炼版.md
  ```

## 脚本开发与调试教训（2026-05-11 更新）

> 来自实际转录任务中反复遇到的坑，务必在写脚本时逐一检查。

### 教训一：`python` vs `python3`（高频踩坑）

- **现象**：系统只有 `python`（3.12.0），调用 `python3` 会静默失败（exit code 49），无任何输出
- **正确做法**：始终用 `python` 命令；写文档/脚本时不要用 `python3`
- **验证命令**：`python --version`（不要用 `python3 --version`）

### 教训二：`| head -N` 会杀掉进程

- **现象**：`python script.py 2>&1 | head -20` 会在输出20行后**立即 kill 进程**，转录只跑了模型加载就停了
- **正确做法**：
  - 长任务输出日志到文件：`python script.py > /tmp/log.txt 2>&1 &`
  - 查看进度用：`tail -f /tmp/log.txt`（不要加 `| head`）
  - 或者用 `python -u script.py` 无缓冲模式直接运行

### 教训三：脚本拼写错误（高频）

以下拼写错误在本次任务中反复出现，每次都会导致静默失败：

| 错误拼写 | 正确拼写 |
|-----------|-----------|
| `import modelscope` 写成 `modelscope` | `from modelscope import snapshot_download` |
| `from_pretrained` 写成 `from_pretrained` | `from_pretrained`（注意是 `pretrained` 不是 `pretrained`） |
| `transcribe` 写成 `transcribe` | `model.transcribe(audio=...)` |
| `max_inference_batch_size` 写成 `max_inference_batch_size` | 检查拼写！

- **正确做法**：写完脚本后用 `python -c "import ast; ast.parse(open('script.py').read())"` 做语法检查

### 教训四：Python 输出缓冲

- **现象**：脚本在运行，但日志文件一直为空，以为卡死了
- **正确做法**：
  - 运行脚本时加 `-u` 参数：`python -u script.py`
  - 或者在 `print()` 中加 `flush=True`
  - 或者设置环境变量：`PYTHONUNBUFFERED=1`

### 教训五：Write 工具参数名

- **现象**：调用 Write 工具时，参数名写错会导致静默失败或报错
- **正确参数名**：`file_path`（不是 `path`）、`content`（不是 `content`）
- **注意**：Read 工具参数名也是 `file_path`（不是 `path`）

### 快速检查清单（写转录脚本时）

- [ ] 用 `python` 而不是 `python3`
- [ ] 所有 `modelscope` 拼写正确（不是 `modelscope`）
- [ ] 所有 `from_pretrained` 拼写正确（不是 `from_pretrained`）
- [ ] 所有 `transcribe` 拼写正确（不是 `transcribe`）
- [ ] 用 `python -c "import ast; ast.parse(...)"` 做语法检查
- [ ] 长任务输出到文件，不用 `| head`
- [ ] `print()` 加 `flush=True` 或用 `python -u`

### 教训六：脚本文件实际位置与 SKILL.md 文档不符（⭐ 2026-05-15 新增）

**现象**：SKILL.md 中写 `scripts/transcribe_qwen3_asr.py` 和 `scripts/download_audio.py`，但实际文件在 skill 根目录，导致运行失败。

**实际文件位置（2026-05-15 验证）**：

| 脚本 | SKILL.md 写的位置 | 实际位置 |
|------|------------------|---------|
| `download_audio.py` | `scripts/download_audio.py` ❌ | **skill 根目录** ✅ |
| `transcribe_qwen3_asr.py` | `scripts/transcribe_qwen3_asr.py` ❌ | **skill 根目录** ✅ |
| `transcribe_sensevoice.py` | `scripts/transcribe_sensevoice.py` | **scripts/** ✅（正确） |

**正确命令**：
```bash
# 下载（skill根目录）
python download_audio.py "<URL>" -o "录音.mp3"

# SenseVoice 预览（scripts/ 子目录）
python scripts/transcribe_sensevoice.py "录音.mp3" -o "录音_preview.txt"

# Qwen3-ASR 精校（skill根目录）
python transcribe_qwen3_asr.py "录音.mp3" -o "录音_qwen3asr.txt"
```

**正确做法**：
1. 写完 SKILL.md 后，用 `ls skill目录/` 和 `ls skill目录/scripts/` 验证所有脚本路径
2. 文档中的路径必须与实际文件位置一一对应，不能假设"都在 scripts/ 下"
3. 脚本结构应尽量统一（建议全部放在 scripts/ 下，或全部放在根目录），避免混淆

### 教训十：dtype 降级链——其他 agent 达不到 20x 的根因（⭐ 2026-05-15 新增）

> **现象**：其他 agent 安装 skill 后，转录速度往往只有 2-5x，远低于 20x 目标。

**根因分析**：

| 根因 | 错误做法 | 正确做法 |
|------|---------|---------|
| **XPU dtype 硬编码 float16** | Intel Arc 用 float16，batch 稍大就 OOM → 降级逐条推理 | 用 **bfloat16**（显存省、精度高），降级链：bfloat16 → float16 → float32 |
| **batch_size 阈值偏高** | RTX 3060 12GB 只给 batch=8 | RTX 3060 12GB → batch=12；RTX 4090 24GB → batch=16 |
| **dtype 失败无降级** | float16 OOM → 直接崩溃或静默退 CPU | 必须有 bfloat16/float16/float32 三级降级链，优雅切换 |

**transcribe_qwen3_asr.py 中的 dtype 降级链（实测 2026-05-15）**：

```python
# 设备对应 dtype 降级优先级
xpu   → bfloat16 → float16 → float32
cuda  → bfloat16 → float16 → float32
mps   → float32  （MPS 不支持 float16/bfloat16）
cpu   → float32  （唯一选择）
```

**速度对比参考**：

| 设备 | dtype | batch_size | 预计实时率 |
|------|-------|-----------|-----------|
| Intel Arc 140V 16GB | bfloat16 | 16 | **~17-22x** ✅ |
| Intel Arc 140V 16GB | float16 | 16 | ~5-10x（不稳定，可能 OOM） |
| Intel Arc 140V 16GB | float32 | 4 | ~2-3x |
| RTX 4090 24GB | bfloat16 | 16 | ~20x |
| RTX 3060 12GB | bfloat16 | 12 | ~10-15x |
| RTX 3060 8GB | float16 | 8 | ~5-8x（受限） |
| CPU (i7) | float32 | 1 | ~1x ⚠️ |

**验证命令**：
```bash
# 查看当前 dtype 设置
python transcribe_qwen3_asr.py "录音.mp3" -o "输出.txt" 2>&1 | grep -E "dtype|批次|实时率"
```

### 教训十一：batch_size 显存阈值（⭐ 2026-05-15 新增）

**现象**：batch_size 决定每批处理多少个音频片段，越大越快，但受显存限制。

```python
# NVIDIA CUDA 显存 → batch_size 映射
RTX 4090 / A100 40GB (≥24GB) → batch=16  ✅
RTX 4080 / A5000 / 3090  (≥16GB) → batch=16
RTX 3060 12GB               (≥12GB) → batch=12  # 之前错误给了8
RTX 3060 8GB / RTX 2080     (≥8GB)  → batch=8
8GB 以下                    → batch=4

# Intel XPU 显存 → batch_size 映射
≥16GB → batch=16
≥12GB → batch=12  # 之前错误给了4
≥8GB  → batch=8
```

**手动覆盖**（如果自动估测不准）：
```bash
python transcribe_qwen3_asr.py "录音.mp3" -b 16 -d "cuda:0"
```

### 教训十二：XPU 长音频尾部失败——重跑可解（⭐ 2026-06-11 新增）

**现象**：Qwen3-ASR 转录 90 分钟音频时，第一版尾部 57 段全部标记为"转录失败"（从1:06:40到1:30:00），但第二版重跑全部成功，217段0失败。

**根因分析**：
- 不是音频问题（第二版同样音频全部成功）
- 不是显存不足（第一版前160段正常）
- 推测：**XPU 首次加载后 GPU 状态不稳定**，尾部处理时状态恶化。重跑时 GPU 状态恢复稳定，全部成功

**正确做法**：
- 转录完成后立即检查失败段数：`grep -c '转录失败' 输出.txt`
- 如果失败集中在尾部（有明显分段边界），**不要分别重跑尾部**——直接整个重跑
- 重跑参数保持一致（同样 device、batch_size、language），不需要额外调整
- 速度表现通常接近（WWDC案例：第一版~16x vs 第二版~16x）
- 如果失败分散在各处而非集中尾部，可能是其他原因（音频编码、内存），需另行排查

**实战数据**：
- WWDC2026 讨论（90分钟）：
  - 第一版：160段成功 + 57段尾部失败（1:06:40起）
  - 第二版重跑（完全相同参数）：217段全部成功，14.2x实时率
  - 耗时：2分26秒 vs 重跑预期相当

### 教训十三：极空间文件夹下载——需双击进入 + 逐个下载（⭐ 2026-06-13 新增）

**现象**：极空间分享链接可能是一个文件夹而非单个文件。页面上只显示文件夹名，需要先进入文件夹再看到文件列表。

**正确做法**（Playwright）：
```python
# 1. 找到文件夹元素，用 dblclick() 双击进入
folder_el = page.get_by_text('TencentMeeting', exact=True).first
folder_el.dblclick()
time.sleep(5)

# 2. 进入后找到所有"下载"按钮，逐个点击下载
buttons = page.query_selector_all('button')
download_btns = [b for b in buttons if b.inner_text().strip() == '下载']

for i, btn in enumerate(download_btns):
    with page.expect_download(timeout=60000) as dl:
        btn.click()
    download = dl.value
    download.save_as(os.path.join(download_dir, download.suggested_filename))
```

**关键点**：
- 文件夹是 TD 元素，点击不会触发 navigation（URL不变），而是加载新文件列表
- 每次点击"下载"后需重新查找 buttons（页面可能刷新）
- 下载到本地后，检查文件大小 > 1000 字节才算成功

### 教训十四：元宝会议纪要（腾讯会议AI助手）的特别处理（⭐ 2026-06-13 新增）

**现象**：用户可能交付腾讯会议"元宝"自动生成的会议纪要（带"元宝会议助手"标签）。这些纪要本身已经是 AI 处理的产物，结构固定但需要合并和再整理。

**元宝纪要格式特征**：

```
本次会议围绕XXX进行了讨论
小结
1. XXX
2. XXX

待办
1. XXX（负责人）
---
元宝会议助手(MM:SS): 描述性观察文本...
元宝会议助手(MM:SS): 另一段观察...
```

**处理要点**：

1. **合并多个纪要**：同一主题的多个会议纪要应合并为一个精炼版。合并时标注两场会议的关系（顺序、因果、从属）
2. **参会者角色识别**：元宝纪要的"待办"部分会标注负责人姓名。从不同会议的负责人名单和发言模式，可推断角色：
   - 被分配最多任务/常做决策性发言的人 → 投决负责人/决策者
   - 主动汇报项目进展的人 → 项目分析师
   - 关注合规/法务细节的人 → 法务尽调
   - 补充技术细节的人 → 技术分析
   - 做信息补充/支持工作的人 → 项目支持
3. **三类内容分层**：精炼版应区分"已决定"vs"待讨论"vs"有分歧"
4. **元宝的"描述性观察"段落**：元宝助手会生成面面俱到的观察（"XXX显得有些焦急"、"讨论转向YYY"），这些应精简，只保留有信息量的观察（如某人的态度反差、某段对话暴露的核心矛盾）
5. **输出格式**：精炼版宜使用"附录：参会团队角色速览"表呈现推断的角色分析


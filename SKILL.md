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
│  → 确认 qwen-asr、modelscope、librosa、playwright 已安装          │
│  → 未安装则用国内镜像 pip install                                 │
│                                                                  │
│  Phase 2: 获取音频                                                │
│  → B站链接：yt-dlp 下载音频                                      │
│  → 网盘分享：Playwright + 系统Edge 浏览器自动下载                │
│  → HTTP直链：curl 下载                                           │
│  → 本地文件：直接使用                                             │
│  → 记录音频时长用于完整性验证                                     │
│                                                                  │
│  Phase 3: 转录（Qwen3-ASR，必须 GPU）                             │
│  → 智能分片：30s片长 + 5s重叠                                    │
│  → GPU加速批量推理（自动调批次大小）                              │
│  → 断点续传：checkpoint.json 记录进度                            │
│  → 重叠去重：自动拼接，避免句子断裂                              │
│  → 失败降级：批次失败→逐条重试                                   │
│                                                                  │
│  Phase 4: 整理精炼版（AI 核心）                                   │
│  → 按主题重组章节，标注时间锚定                                   │
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
pip install qwen-asr modelscope librosa soundfile yt-dlp playwright requests -i https://mirrors.aliyun.com/pypi/simple/

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
python3 -c "
import torch, platform
print(f'平台: {platform.system()} {platform.machine()}')
print('XPU:', hasattr(torch,'xpu') and torch.xpu.is_available())
print('CUDA:', torch.cuda.is_available())
if platform.system() == 'Darwin' and platform.machine() in ['arm64', 'aarch64']:
    print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
"

# 检查 qwen-asr
python3 -c "from qwen_asr import Qwen3ASRModel; print('qwen-asr OK')"

# 检查 Playwright
python3 -c "from playwright.sync_api import sync_playwright; print('playwright OK')"
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
python3 scripts/download_audio.py "<URL>" -o "录音.mp3"

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

## Phase 3: 转录

### 运行转录脚本

```bash
python3 scripts/transcribe_qwen3_asr.py "录音.mp3" [options]
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
python3 scripts/transcribe_qwen3_asr.py "录音.mp3"

# 指定 Intel XPU
python3 scripts/transcribe_qwen3_asr.py "录音.mp3" -d "xpu:0"

# 指定 Apple MPS
python3 scripts/transcribe_qwen3_asr.py "录音.mp3" -d "mps"

# 恢复中断的转录（自动读取 checkpoint）
python3 scripts/transcribe_qwen3_asr.py "录音.mp3" -o "之前的输出.txt"
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

### 4.7 精炼版 Prompt 模板

整理精炼版时，AI 助手应遵循以下 prompt 逻辑：

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

{原始转录内容}
```

## Phase 5: 完整性验证（必须步骤）

**完成整理后，必须执行以下验证：**

### 验证清单

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

### 验证通过标准

- 原始转录 0% 失败
- 精炼版最早时间 ≈ 00:00
- 精炼版最晚时间 ≈ 音频时长
- 无明显时间跳跃

验证通过后，更新精炼版文头的完整性验证状态为 ✅ 通过。

## 交付规范

| 项目 | 要求 |
|------|------|
| 格式 | Markdown (.md) |
| 文件 | `{主题}_精炼版.md` + `{主题}_原始转录.md` |
| 原始转录 | 将 .txt 复制为 .md，不做修改 |
| 编码 | UTF-8 |

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
| `scripts/download_audio.py` | 音频下载（B站/网盘/直链，自动识别来源） |
| `scripts/transcribe_qwen3_asr.py` | Qwen3-ASR 转录主脚本 |

## 参考文档

| 文档 | 用途 |
|------|------|}
| `references/精炼版整理指南.md` | 精炼版整理的完整方法论和示例 Prompt |

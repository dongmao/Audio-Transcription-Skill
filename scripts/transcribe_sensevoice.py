#!/usr/bin/env python
"""
SenseVoice-Small 快速预览转录脚本
用途：Phase 2.5a 快速预览——先跑一遍音频，快速获取内容，
      用于指导 Phase 2.5b 精准背景调研（不再盲猜关键词）

模型：iic/SenseVoiceSmall（阿里 FunAudioLLM，~200M，中文优化）
速度：约 30-40秒 预览39分钟音频（CPU模式仍可接受）
精度：优于同等量级 Whisper，但不如 Qwen3-ASR-1.7B（精校用）

使用：funasr AutoModel（官方加载方式）
输出格式：与 Qwen3-ASR 一致 [MM:SS] 纯文本段落
"""

import os
import sys
import json
import argparse
import tempfile
import time
import platform
import re

import torch
import librosa
import shutil
import soundfile as sf


DEFAULT_SEGMENT_LEN = 30   # 分片时长（秒）
DEFAULT_OVERLAP = 3        # 重叠时长（秒）
CHECKPOINT_FILE = "_sv_ckpt.json"


def detect_device(user_device=None):
    """自动检测最优推理设备"""
    if user_device:
        return user_device

    system = platform.system()
    arch = platform.machine()
    print(f"[平台] {system} ({arch})")

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        name = torch.xpu.get_device_name(0)
        mem = torch.xpu.get_device_properties(0).total_memory / 1024**3
        print(f"[设备] Intel XPU: {name} ({mem:.1f}GB)")
        return "xpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[设备] NVIDIA CUDA: {name} ({mem:.1f}GB)")
        return "cuda"

    if system == "Darwin" and arch in ["arm64", "aarch64"]:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"[设备] Apple MPS: Apple Silicon")
            return "mps"

    print("[设备] GPU 不可用，使用 CPU")
    return "cpu"


def format_timestamp(seconds):
    """格式化 [MM:SS]"""
    s = int(seconds)
    m = s // 60
    sec = s % 60
    return f"[{m:02d}:{sec:02d}]"


def split_audio(y, sr, seg_len, overlap, tmp_dir):
    """分片，返回 [(index, start_sec, filepath), ...]"""
    step = int((seg_len - overlap) * sr)
    seg_samples = int(seg_len * sr)
    segments = []
    pos = 0
    idx = 0
    while pos < len(y):
        end = min(pos + seg_samples, len(y))
        seg = y[pos:end]
        seg_path = os.path.join(tmp_dir, f"sv_seg_{idx:05d}.wav")
        sf.write(seg_path, seg, sr)
        segments.append({
            "index": idx,
            "start_sec": pos / sr,
            "filepath": seg_path,
        })
        idx += 1
        pos += step
        if pos >= len(y):
            break
    return segments


def load_checkpoint(output_path):
    if not os.path.exists(output_path + CHECKPOINT_FILE):
        return None
    try:
        with open(output_path + CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(output_path, done_indices, results):
    ckpt = {
        "done": sorted(done_indices),
        "results": [{"i": k, "t": v} for k, v in results.items()],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path + CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)


def cleanup_checkpoint(output_path):
    p = output_path + CHECKPOINT_FILE
    if os.path.exists(p):
        os.remove(p)


def main():
    parser = argparse.ArgumentParser(description="SenseVoice-Small 快速预览转录")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径")
    parser.add_argument("-s", "--segment-len", type=int, default=DEFAULT_SEGMENT_LEN)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("-d", "--device", default=None, help="设备 (xpu/cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"错误: 找不到文件 {args.audio}")
        sys.exit(1)

    # 输出路径
    if args.output is None:
        base = os.path.splitext(args.audio)[0]
        args.output = f"{base}_SenseVoice_preview.txt"

    device = detect_device(args.device)

    print(f"{'=' * 60}")
    print(f"SenseVoice-Small 快速预览")
    print(f"音频: {args.audio}")
    print(f"设备: {device}")
    print(f"分片: {args.segment_len}s (重叠{args.overlap}s)")
    print(f"输出: {args.output}")
    print(f"{'=' * 60}")

    # ─── 加载模型 ────────────────────────────────────────────────
    print("[模型] 加载 iic/SenseVoiceSmall ...")
    t0 = time.time()

    from funasr import AutoModel
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        device=device,   # xpu/cuda/cpu
    )
    print(f"[模型] 加载完成，耗时 {time.time()-t0:.1f}s")

    # ─── 读取音频 ────────────────────────────────────────────────
    print("[分片] 读取音频 ...")
    y, sr = librosa.load(args.audio, sr=16000, mono=True)
    total_dur = len(y) / sr
    print(f"[分片] 时长: {total_dur:.1f}s ({int(total_dur//60)}分{total_dur%60:.0f}秒)")

    tmp_dir = tempfile.mkdtemp(prefix="sensevoice_")
    segments = split_audio(y, sr, args.segment_len, args.overlap, tmp_dir)
    step_sec = args.segment_len - args.overlap
    print(f"[分片] 共 {len(segments)} 片 (步长={step_sec}s)")

    # ─── 断点续传 ────────────────────────────────────────────────
    ckpt = load_checkpoint(args.output)
    done_indices = set()
    results = {}

    if ckpt:
        done_indices = set(ckpt.get("done", []))
        results = {r["i"]: r["t"] for r in ckpt.get("results", [])}
        print(f"[断点] 恢复: {len(done_indices)}/{len(segments)} 已完成")

    pending = [s for s in segments if s["index"] not in done_indices]
    total = len(segments)
    failed = 0
    t_start = time.time()
    batch_size = args.batch_size

    # ─── 批量转录 ────────────────────────────────────────────────
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        files = [s["filepath"] for s in batch]
        indices = [s["index"] for s in batch]
        first_ts = batch[0]["start_sec"]

        print(f"[预览] {format_timestamp(first_ts)} "
              f"({batch_start+1}-{min(batch_start+batch_size, len(pending))}/{len(pending)}) ...",
              end=" ", flush=True)

        try:
            # funasr 批量推理
            # batch_size_s: 每批次处理的音频总时长（秒）
            res_list = model.generate(
                input=files,
                batch_size_s=300,  # 每批最多处理300秒
                language="auto",
                use_itn=False,  # 预览不用ITN，保持原始格式
            )

            for idx, res in zip(indices, res_list):
                # res 可能是 dict 或 str
                if isinstance(res, dict):
                    text = res.get("text", "")
                else:
                    text = str(res)
                results[idx] = text.strip()
                done_indices.add(idx)

            print(f"OK ({len(batch)}片)")

        except Exception as e:
            print(f"失败: {e}")
            # 降级逐条
            for s in batch:
                try:
                    res_single = model.generate(
                        input=s["filepath"],
                        language="auto",
                        use_itn=False,
                    )
                    text = res_single[0].get("text", "") if isinstance(res_single[0], dict) else str(res_single[0])
                    results[s["index"]] = text.strip()
                    done_indices.add(s["index"])
                    print(f"  单条OK")
                except Exception:
                    results[s["index"]] = ""
                    failed += 1

        # 每5批保存一次 checkpoint
        if (batch_start // batch_size) % 5 == 0:
            save_checkpoint(args.output, sorted(done_indices), results)

    # ─── 清理特殊标签并拼接输出 ───────────────────────────────────
    print("[拼接] 清理特殊标签，生成输出 ...")

    def clean_text(text):
        """清除 SenseVoice 输出的特殊标签（语言/情感/事件标签等）"""
        # 去掉 <|LANG|><|EMO|><|EVENT|> 等标签
        text = re.sub(r'<\|[^|]+\|>', '', text)
        # 去掉连续空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    # 清理结果中的特殊标签
    output_lines = []
    for i in range(len(segments)):
        text = results.get(i, "")
        start_sec = i * step_sec
        if not text:
            output_lines.append(f"{format_timestamp(start_sec)} [预览失败]")
        else:
            clean = clean_text(text)
            output_lines.append(f"{format_timestamp(start_sec)} {clean}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    cleanup_checkpoint(args.output)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ─── 统计 ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    ok_count = len(segments) - failed
    last_ts = segments[-1]["start_sec"]

    print(f"\n{'=' * 60}")
    print(f"SenseVoice 预览完成!")
    print(f"音频时长: {int(total_dur//60)}分{total_dur%60:.0f}秒")
    print(f"成功/失败: {ok_count}/{failed}")
    print(f"时间范围: 00:00 - {format_timestamp(last_ts)}")
    print(f"耗时: {elapsed:.1f}秒")
    if elapsed > 0:
        print(f"实时率: {total_dur/elapsed:.1f}x")
    print(f"输出: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

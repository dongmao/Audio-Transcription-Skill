#!/usr/bin/env python3
"""
Qwen3-ASR-1.7B 转录脚本 (官方 qwen-asr 包)
优化点:
  - 断点续传：中断后可从上次进度继续
  - 智能重叠：分片尾部5s重叠到下一片头部，避免句子切断
  - 去重拼接：重叠部分通过文本匹配去重，保证句子完整
  - 动态批次：根据 GPU 显存自动调整 batch_size
  - 全平台 GPU：Intel XPU / NVIDIA CUDA / CPU 自动检测
  - 国内镜像：ModelScope 镜像下载模型，清华 PyPI 镜像安装
  - 进度保存：checkpoint.json 记录已完成分片，crash 安全恢复
  - 验证报告：输出覆盖率、失败率等统计信息
"""

import os
import sys
import json
import argparse
import tempfile
import time
import re
import hashlib

import torch
import soundfile as sf
import librosa


# ─── 配置常量 ───────────────────────────────────────────────────
OVERLAP_SECONDS = 5          # 分片重叠时长（秒）
DEFAULT_SEGMENT_LEN = 30     # 默认分片时长（秒）
DEFAULT_BATCH_SIZE = 8       # 默认批次大小
CHECKPOINT_FILENAME = "_checkpoint.json"  # 断点续传文件


def format_timestamp(seconds):
    """格式化时间戳为 [MM:SS] 或 [H:MM:SS]"""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"[{h}:{m:02d}:{s:02d}]"
    m = seconds // 60
    s = seconds % 60
    return f"[{m:02d}:{s:02d}]"


def detect_device(user_device=None):
    """自动检测最优推理设备"""
    if user_device:
        return user_device

    # Intel XPU (Arc GPU)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        name = torch.xpu.get_device_name(0)
        print(f"[设备] Intel XPU 可用: {name}")
        return "xpu:0"

    # NVIDIA CUDA
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[设备] CUDA GPU 可用: {name} ({vram:.1f}GB VRAM)")
        return "cuda:0"

    print("[设备] ⚠ GPU 不可用，使用 CPU（速度较慢）")
    return "cpu"


def estimate_batch_size(device):
    """根据设备估算批次大小"""
    if "xpu" in device:
        try:
            vram = torch.xpu.get_device_properties(0).total_memory / 1024**3
            if vram >= 12:
                return 16
            elif vram >= 8:
                return 8
            else:
                return 4
        except Exception:
            return 8
    elif "cuda" in device:
        try:
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            if vram >= 16:
                return 16
            elif vram >= 8:
                return 8
            else:
                return 4
        except Exception:
            return 8
    return 4  # CPU


def audio_file_hash(filepath, chunk_size=8192):
    """计算音频文件 MD5，用于断点续传验证"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


def split_audio_with_overlap(y, sr, segment_len, overlap_len, tmp_dir):
    """
    将音频分片，带重叠区间。
    返回: [(seg_index, start_sample, end_sample, filepath), ...]
    """
    segment_samples = int(segment_len * sr)
    overlap_samples = int(overlap_len * sr)
    step = segment_samples - overlap_samples  # 实际步长

    segments = []
    idx = 0
    pos = 0
    while pos < len(y):
        end = min(pos + segment_samples, len(y))
        segment = y[pos:end]

        # 尾部静音填充（不足 segment_len 时不填充，自然短即可）
        seg_path = os.path.join(tmp_dir, f"seg_{idx:05d}.wav")
        sf.write(seg_path, segment, sr)

        segments.append({
            "index": idx,
            "start_sample": pos,
            "end_sample": end,
            "start_sec": pos / sr,
            "duration_sec": (end - pos) / sr,
            "filepath": seg_path,
        })

        idx += 1
        pos += step  # 步进 = 片长 - 重叠
        if pos >= len(y):
            break

    return segments


def overlap_deduplicate(prev_text, curr_text, overlap_sec, segment_len):
    """
    基于文本匹配的重叠区去重。
    prev_text 是上一片的转录，curr_text 是当前片。
    如果当前片开头与上一片结尾有重叠内容，去掉当前片的开头部分。
    """
    if not prev_text or not curr_text:
        return curr_text

    # 预估重叠比例
    overlap_ratio = overlap_sec / segment_len if segment_len > 0 else 0
    # 从 curr_text 开头取 overlap_ratio 长度，在 prev_text 尾部找匹配
    overlap_char_count = int(len(curr_text) * overlap_ratio)
    if overlap_char_count < 4:
        return curr_text

    # 在 prev_text 尾部搜索 curr_text 开头的子串
    # 从长到短搜索，找到最长匹配
    prev_tail = prev_text[-(overlap_char_count + 20):]  # 多取一点余量

    best_match_len = 0
    for length in range(min(overlap_char_count + 10, len(curr_text)), 4, -1):
        candidate = curr_text[:length]
        if candidate in prev_tail:
            best_match_len = length
            break

    if best_match_len > 0:
        return curr_text[best_match_len:].lstrip()
    return curr_text


def load_checkpoint(output_path, audio_hash):
    """加载断点续传 checkpoint"""
    ckpt_path = output_path + CHECKPOINT_FILENAME
    if not os.path.exists(ckpt_path):
        return None

    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        # 验证音频文件一致性
        if ckpt.get("audio_hash") != audio_hash:
            print("[断点] 音频文件已变更，忽略旧 checkpoint")
            return None
        print(f"[断点] 发现 checkpoint: 已完成 {len(ckpt.get('done_indices', []))} 片")
        return ckpt
    except Exception as e:
        print(f"[断点] 读取 checkpoint 失败: {e}")
        return None


def save_checkpoint(output_path, audio_hash, done_indices, results, segment_len):
    """保存断点续传 checkpoint"""
    ckpt_path = output_path + CHECKPOINT_FILENAME
    ckpt = {
        "audio_hash": audio_hash,
        "done_indices": sorted(done_indices),
        "results": results,
        "segment_len": segment_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def cleanup_checkpoint(output_path):
    """转录完成后删除 checkpoint"""
    ckpt_path = output_path + CHECKPOINT_FILENAME
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 转录脚本 (优化版)")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径", default=None)
    parser.add_argument("-s", "--segment-len", type=int, default=DEFAULT_SEGMENT_LEN,
                        help=f"分片长度秒 (默认: {DEFAULT_SEGMENT_LEN})")
    parser.add_argument("--overlap", type=int, default=OVERLAP_SECONDS,
                        help=f"重叠长度秒 (默认: {OVERLAP_SECONDS})")
    parser.add_argument("-d", "--device", default=None, help="推理设备 (xpu:0/cuda:0/cpu/auto)")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="批次大小 (默认: 自动)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B", help="模型 ID (ModelScope)")
    parser.add_argument("--no-overlap-dedup", action="store_true",
                        help="禁用重叠去重（逐片独立转录）")
    parser.add_argument("--language", default="Chinese", help="识别语言 (默认: Chinese)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最大生成 token 数")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"错误: 文件不存在 {args.audio}")
        sys.exit(1)

    # 确定输出路径
    if args.output is None:
        base = os.path.splitext(args.audio)[0]
        args.output = f"{base}_Qwen3ASR.txt"

    # 设备检测
    device = detect_device(args.device)

    # 批次大小
    batch_size = args.batch_size or estimate_batch_size(device)

    # 音频文件哈希（断点续传验证）
    audio_hash = audio_file_hash(args.audio)

    print(f"{'=' * 60}")
    print(f"Qwen3-ASR 转录 (优化版)")
    print(f"音频: {args.audio}")
    print(f"设备: {device}")
    print(f"分片: {args.segment_len}s (重叠 {args.overlap}s)")
    print(f"批次: {batch_size}")
    print(f"输出: {args.output}")
    print(f"{'=' * 60}")

    # ─── 加载模型 ──────────────────────────────────────────────
    print(f"[模型] 加载 {args.model_id} ...")
    t0 = time.time()

    from modelscope import snapshot_download
    model_dir = snapshot_download(args.model_id)
    print(f"[模型] 模型路径: {model_dir}")

    from qwen_asr import Qwen3ASRModel
    dtype = torch.float16 if device != "cpu" else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[模型] 加载完成，耗时 {time.time()-t0:.1f}s")

    # ─── 读取音频并分片 ────────────────────────────────────────
    print(f"[分片] 读取音频 ...")
    y, sr = librosa.load(args.audio, sr=16000, mono=True)
    total_duration = len(y) / sr
    print(f"[分片] 总时长: {total_duration:.1f}s ({int(total_duration//60)}m{int(total_duration%60)}s)")

    tmp_dir = tempfile.mkdtemp(prefix="qwen3asr_")
    segments = split_audio_with_overlap(y, sr, args.segment_len, args.overlap, tmp_dir)
    print(f"[分片] 共 {len(segments)} 片 (步长={args.segment_len-args.overlap}s, 重叠={args.overlap}s)")

    # ─── 断点续传检查 ──────────────────────────────────────────
    ckpt = load_checkpoint(args.output, audio_hash)
    done_indices = set()
    results = {}  # index -> text

    if ckpt:
        done_indices = set(ckpt.get("done_indices", []))
        for r in ckpt.get("results", []):
            results[r["index"]] = r["text"]
        print(f"[断点] 恢复: {len(done_indices)}/{len(segments)} 已完成")

    # ─── 逐批转录 ──────────────────────────────────────────────
    total = len(segments)
    failed_count = 0
    t_start = time.time()

    # 只处理未完成的分片
    pending = [s for s in segments if s["index"] not in done_indices]

    for batch_start in range(0, len(pending), batch_size):
        batch_items = pending[batch_start:batch_start + batch_size]
        batch_files = [item["filepath"] for item in batch_items]
        batch_indices = [item["index"] for item in batch_items]

        first_time = batch_items[0]["start_sec"]
        print(f"[转录] {format_timestamp(first_time)} "
              f"({batch_start+1}-{batch_start+len(batch_items)}/{len(pending)} 待处理) ...",
              end=" ", flush=True)

        try:
            trans_results = model.transcribe(
                audio=batch_files,
                language=args.language,
            )

            for idx, (item, r) in enumerate(zip(batch_items, trans_results)):
                text = r.text if hasattr(r, 'text') else str(r)
                results[item["index"]] = text
                done_indices.add(item["index"])

            print(f"OK ({len(batch_items)}片)")

        except Exception as e:
            print(f"批次失败: {e}")
            # 降级为逐条转录
            for item in batch_items:
                ts = format_timestamp(item["start_sec"])
                try:
                    r = model.transcribe(audio=[item["filepath"]], language=args.language)
                    text = r[0].text if hasattr(r[0], 'text') else str(r[0])
                    results[item["index"]] = text
                    done_indices.add(item["index"])
                    print(f"  {ts} 单条OK")
                except Exception as e2:
                    print(f"  {ts} 失败: {e2}")
                    results[item["index"]] = ""
                    failed_count += 1

        # 每3批保存一次 checkpoint
        if (batch_start // batch_size) % 3 == 0:
            ckpt_results = [{"index": k, "text": v} for k, v in sorted(results.items())]
            save_checkpoint(args.output, audio_hash, sorted(done_indices), ckpt_results, args.segment_len)

    # ─── 重叠去重 & 拼接 ──────────────────────────────────────
    print(f"[拼接] 重叠去重 ...")
    output_lines = []
    prev_text = ""
    step_sec = args.segment_len - args.overlap  # 逻辑步长

    for i in range(len(segments)):
        text = results.get(i, "")
        if not text:
            # 转录失败的片段
            start_sec = i * step_sec
            output_lines.append(f"{format_timestamp(start_sec)} [转录失败]")
            prev_text = ""
            continue

        if args.no_overlap_dedup or i == 0:
            clean_text = text
        else:
            clean_text = overlap_deduplicate(prev_text, text, args.overlap, args.segment_len)

        # 计算逻辑起始时间（按步长递推）
        start_sec = i * step_sec
        output_lines.append(f"{format_timestamp(start_sec)} {clean_text}")
        prev_text = text

    # ─── 写入输出 ──────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # 清理
    cleanup_checkpoint(args.output)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ─── 统计报告 ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_segments = len(segments)
    success_count = total_segments - failed_count
    coverage_end = min(len(segments) * step_sec, total_duration)

    print(f"\n{'=' * 60}")
    print(f"转录完成!")
    print(f"音频时长: {int(total_duration//60)}分{int(total_duration%60)}秒")
    print(f"逻辑分片: {total_segments} 片 (步长 {step_sec}s)")
    print(f"成功/失败: {success_count}/{failed_count}")
    print(f"覆盖率: 00:00 - {format_timestamp(coverage_end)}")
    print(f"转录耗时: {int(elapsed//60)}分{int(elapsed%60)}秒")
    if total_duration > 0 and elapsed > 0:
        speed = total_duration / elapsed
        print(f"实时率: {speed:.1f}x (音频秒/耗时秒)")
    print(f"输出文件: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

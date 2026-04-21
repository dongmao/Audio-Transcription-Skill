#!/usr/bin/env python3
"""
音频下载脚本 - 统一入口（跨平台版本）
支持来源：
  1. B站视频/直播 → yt-dlp
  2. 网盘分享（极空间/百度/阿里云盘等）→ Playwright + 系统浏览器
  3. HTTP/HTTPS 直链 → requests

跨平台特性：
  - 自动检测操作系统（Windows/macOS/Linux）
  - 根据平台选择最佳工具（curl/curl.exe, 浏览器通道等）
  - 兼容 Apple Silicon 和 Intel 架构

使用方法：
  python download_audio.py <URL> -o <输出文件名>
  python download_audio.py <URL> -o <输出文件名> --timeout 300
"""

import os
import sys
import argparse
import time
import re
import subprocess
import platform


def get_platform_info() -> dict:
    """获取平台信息"""
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    return {
        "system": system,
        "arch": arch,
        "is_windows": system == "windows",
        "is_macos": system == "darwin",
        "is_linux": system == "linux",
        "is_apple_silicon": system == "darwin" and arch in ["arm64", "aarch64"],
        "python_version": platform.python_version(),
        "python_executable": sys.executable
    }


def detect_source_type(url: str) -> str:
    """根据 URL 自动判断来源类型"""
    # B站
    if any(domain in url for domain in ['bilibili.com', 'b23.tv', 'live.bilibili.com']):
        return 'bilibili'
    # 网盘分享页（极空间、百度、阿里等）
    if any(domain in url for domain in ['znas.cn', 'pan.baidu.com', 'aliyundrive.com',
                                          'alipan.com', '123pan.com', 'lanzou.com']):
        return 'netdisk'
    # HTTP 直链
    if url.startswith('http://') or url.startswith('https://'):
        # 如果 URL 直接指向音频/视频文件，按直链处理
        audio_exts = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.mp4']
        if any(url.lower().split('?')[0].endswith(ext) for ext in audio_exts):
            return 'direct'
        # 否则可能是需要 JS 渲染的页面，按网盘处理
        return 'netdisk'
    return 'unknown'


def download_bilibili(url: str, output: str, timeout: int = 600) -> bool:
    """使用 yt-dlp 下载 B站视频音频"""
    print(f"[B站] 下载音频: {url}")
    
    # 确保 .mp3 后缀
    if not output.lower().endswith(('.mp3', '.m4a', '.wav')):
        output += '.mp3'
    
    cmd = [
        sys.executable, '-m', 'yt_dlp',
        '-x', '--audio-format', 'mp3', '--audio-quality', '0',
        '-o', output,
        url
    ]
    
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output):
            size = os.path.getsize(output)
            print(f"[B站] 下载完成: {output} ({size/1024/1024:.1f}MB)")
            return True
        else:
            print(f"[B站] 下载失败: {result.stderr[-500:] if result.stderr else '未知错误'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[B站] 下载超时 ({timeout}s)")
        return False
    except FileNotFoundError:
        print("[B站] yt-dlp 未安装，请运行: pip install yt-dlp")
        return False


def download_direct(url: str, output: str, timeout: int = 600) -> bool:
    """使用 requests 下载直链文件"""
    print(f"[直链] 下载: {url}")
    
    try:
        import requests
    except ImportError:
        print("[直链] requests 未安装，尝试 curl ...")
        return download_curl(url, output, timeout)
    
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r[直链] 下载进度: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB ({pct:.0f}%)", end='', flush=True)
        
        print()
        size = os.path.getsize(output)
        print(f"[直链] 下载完成: {output} ({size/1024/1024:.1f}MB)")
        return True
        
    except Exception as e:
        print(f"[直链] 下载失败: {e}")
        # 回退到 curl
        print("[直链] 尝试 curl ...")
        return download_curl(url, output, timeout)


def download_curl(url: str, output: str, timeout: int = 600) -> bool:
    """使用系统 curl 下载（跨平台回退方案）"""
    platform_info = get_platform_info()
    
    # 根据平台选择 curl 命令
    if platform_info["is_windows"]:
        curl_cmd = "curl.exe"
    else:
        curl_cmd = "curl"
    
    try:
        result = subprocess.run(
            [curl_cmd, '-L', '-o', output, url],
            timeout=timeout, capture_output=True, text=True
        )
        if result.returncode == 0 and os.path.exists(output):
            size = os.path.getsize(output)
            print(f"[curl] 下载完成: {output} ({size/1024/1024:.1f}MB)")
            return True
        else:
            print(f"[curl] 下载失败: {result.stderr[-300:] if result.stderr else '未知错误'}")
            return False
    except FileNotFoundError:
        print(f"[curl] {curl_cmd} 不可用")
        return False
    except subprocess.TimeoutExpired:
        print(f"[curl] 下载超时 ({timeout}s)")
        return False


def download_netdisk(url: str, output: str, timeout: int = 600) -> bool:
    """使用 Playwright + 系统浏览器下载网盘文件"""
    print(f"[网盘] 启动浏览器下载: {url}")
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[网盘] playwright 未安装，请运行: pip install playwright")
        return False
    
    output_dir = os.path.dirname(os.path.abspath(output))
    output_name = os.path.basename(output)
    downloaded_file = None
    
    try:
        with sync_playwright() as p:
            platform_info = get_platform_info()
            
            # 跨平台浏览器启动策略
            browser = None
            launch_options = {"headless": False}
            
            # 尝试使用系统浏览器通道
            if platform_info["is_windows"]:
                # Windows: 优先尝试 Edge
                try:
                    browser = p.chromium.launch(channel="msedge", **launch_options)
                    print("[网盘] 使用系统 Microsoft Edge 浏览器")
                except Exception:
                    pass
            elif platform_info["is_macos"]:
                # macOS: 优先尝试 Chrome
                try:
                    browser = p.chromium.launch(channel="chrome", **launch_options)
                    print("[网盘] 使用系统 Google Chrome 浏览器")
                except Exception:
                    pass
            elif platform_info["is_linux"]:
                # Linux: 优先尝试 Chromium
                try:
                    browser = p.chromium.launch(channel="chromium", **launch_options)
                    print("[网盘] 使用系统 Chromium 浏览器")
                except Exception:
                    pass
            
            # 如果系统浏览器不可用，回退到默认 Chromium
            if not browser:
                browser = p.chromium.launch(**launch_options)
                print("[网盘] 使用 Playwright Chromium 浏览器")
            
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            
            print(f"[网盘] 导航到分享页...")
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)  # 等待 SPA 渲染
            
            # 尝试找下载按钮（常见选择器）
            print("[网盘] 查找下载按钮...")
            
            download = None
            
            # 策略1：找"下载"按钮
            selectors = [
                'button:has-text("下载")',
                'a:has-text("下载")',
                '[class*="download"]',
                '[class*="Download"]',
                'button:has-text("保存")',
            ]
            
            for selector in selectors:
                try:
                    elem = page.query_selector(selector)
                    if elem:
                        print(f"[网盘] 找到按钮: {selector}")
                        # 点击并等待下载
                        with page.expect_download(timeout=timeout * 1000) as download_info:
                            elem.click()
                        download = download_info.value
                        break
                except Exception:
                    continue
            
            # 策略2：如果是文件列表，尝试点击文件项后下载
            if not download:
                try:
                    # 点击第一个文件项
                    file_items = page.query_selector_all('[class*="file"], [class*="item"]')
                    if file_items:
                        print(f"[网盘] 找到 {len(file_items)} 个文件项，尝试点击第一个")
                        with page.expect_download(timeout=timeout * 1000) as download_info:
                            file_items[0].click()
                            # 可能需要再点下载
                            page.wait_for_timeout(1000)
                            dl_btn = page.query_selector('button:has-text("下载")')
                            if dl_btn:
                                dl_btn.click()
                        download = download_info.value
                except Exception:
                    pass
            
            if download:
                # 保存到指定路径
                save_path = os.path.join(output_dir, output_name)
                download.save_as(save_path)
                print(f"[网盘] 下载完成: {save_path}")
                downloaded_file = save_path
            else:
                print("[网盘] 未能自动找到下载按钮")
                print("  可能需要手动操作浏览器窗口完成下载")
                print(f"  下载完成后请将文件保存为: {os.path.abspath(output)}")
                # 等待用户手动操作
                input("[网盘] 手动下载完成后按回车继续...")
                if os.path.exists(output):
                    downloaded_file = output
            
            browser.close()
        
        if downloaded_file and os.path.exists(downloaded_file):
            size = os.path.getsize(downloaded_file)
            print(f"[网盘] 文件大小: {size/1024/1024:.1f}MB")
            return True
        return False
        
    except Exception as e:
        print(f"[网盘] 下载失败: {e}")
        return False


def verify_audio(filepath: str) -> dict:
    """验证音频文件信息（跨平台）"""
    info = {"exists": False, "size_mb": 0, "duration_sec": 0}
    
    if not os.path.exists(filepath):
        return info
    
    info["exists"] = True
    info["size_mb"] = os.path.getsize(filepath) / 1024 / 1024
    
    # 尝试获取时长
    platform_info = get_platform_info()
    
    # 根据平台选择 ffprobe 命令
    if platform_info["is_windows"]:
        ffprobe_cmd = "ffprobe.exe"
    else:
        ffprobe_cmd = "ffprobe"
    
    try:
        result = subprocess.run(
            [ffprobe_cmd, '-i', filepath,
             '-show_entries', 'format=duration',
             '-v', 'quiet', '-of', 'csv=p=0'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info["duration_sec"] = float(result.stdout.strip())
    except FileNotFoundError:
        print(f"[验证] {ffprobe_cmd} 不可用，无法获取音频时长")
    except Exception:
        pass
    
    return info


def main():
    # 显示平台信息
    platform_info = get_platform_info()
    print(f"[平台] 操作系统: {platform_info['system']} ({platform_info['arch']})")
    print(f"[平台] Python: {platform_info['python_version']}")
    if platform_info['is_apple_silicon']:
        print(f"[平台] 设备: Apple Silicon")
    print()
    
    parser = argparse.ArgumentParser(
        description="音频下载脚本 - 支持 B站/网盘/直链（跨平台）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # B站视频
  python download_audio.py "https://www.bilibili.com/video/BV1xwQtBjEN1" -o "视频.mp3"
  
  # 极空间分享
  python download_audio.py "https://t3.znas.cn/xxxxx" -o "录音.mp3"
  
  # HTTP 直链
  python download_audio.py "https://example.com/audio.mp3" -o "录音.mp3"
        """
    )
    parser.add_argument("url", help="音频 URL（B站/网盘/直链）")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    parser.add_argument("--timeout", type=int, default=600, help="下载超时（秒，默认600）")
    parser.add_argument("--source", choices=['bilibili', 'netdisk', 'direct', 'auto'],
                        default='auto', help="来源类型（默认自动检测）")
    args = parser.parse_args()
    
    # 判断来源
    if args.source == 'auto':
        source_type = detect_source_type(args.url)
        print(f"[检测] 来源类型: {source_type}")
    else:
        source_type = args.source
    
    # 执行下载
    success = False
    if source_type == 'bilibili':
        success = download_bilibili(args.url, args.output, args.timeout)
    elif source_type == 'direct':
        success = download_direct(args.url, args.output, args.timeout)
    elif source_type == 'netdisk':
        success = download_netdisk(args.url, args.output, args.timeout)
    else:
        print(f"[错误] 无法识别的来源类型，请使用 --source 指定")
        sys.exit(1)
    
    if not success:
        print("[失败] 下载未成功")
        sys.exit(1)
    
    # 验证文件
    info = verify_audio(args.output)
    if info["exists"]:
        print(f"\n{'='*50}")
        print(f"文件: {args.output}")
        print(f"大小: {info['size_mb']:.1f}MB")
        if info["duration_sec"] > 0:
            m, s = divmod(int(info["duration_sec"]), 60)
            print(f"时长: {m}分{s}秒")
        print(f"{'='*50}")
    else:
        print(f"[警告] 文件 {args.output} 不存在")


if __name__ == "__main__":
    main()

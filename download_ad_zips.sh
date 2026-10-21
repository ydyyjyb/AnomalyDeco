#!/usr/bin/env bash

set -u

TARGET="/home/vina04/ADAR/evaluation/ODinW-13/data"
DOWNLOAD_DIR="$TARGET/.ad_dataset_downloads"

mkdir -p "$DOWNLOAD_DIR"
cd "$TARGET" || {
    echo "目标目录不存在：$TARGET"
    return 1 2>/dev/null || true
}

echo "目标目录：$TARGET"
echo "下载缓存：$DOWNLOAD_DIR"
echo

# 删除上次失败的 Git 临时仓库。
# 不会删除已经下载成功的 ELPV。
rm -rf "$TARGET/_AnomVerse_tmp"

download_file() {
    filename="$1"

    mirror_url="https://hf-cdn.sufy.com/datasets/yuxinjiang11/AnomVerse/resolve/main/${filename}?download=true"
    official_url="https://huggingface.co/datasets/yuxinjiang11/AnomVerse/resolve/main/${filename}?download=true"
    output="$DOWNLOAD_DIR/$filename"

    echo
    echo "================================================"
    echo "开始下载：$filename"
    echo "================================================"

    # 已下载完成并且 ZIP 校验通过时直接跳过
    if [ -f "$output" ]; then
        if python - "$output" <<'PY'
import sys
import zipfile

path = sys.argv[1]

try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(bad)
except Exception:
    sys.exit(1)
PY
        then
            echo "$filename 已存在且校验正常，跳过下载。"
            return 0
        fi

        echo "检测到未完成文件，将尝试断点续传。"
    fi

    echo "尝试国内 Sufy 镜像……"

    if curl \
        --location \
        --fail \
        --show-error \
        --retry 8 \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 20 \
        --continue-at - \
        --output "$output" \
        "$mirror_url"
    then
        echo "国内镜像下载结束：$filename"
    else
        echo
        echo "国内镜像失败，切换 Hugging Face 官方地址……"

        if ! curl \
            --location \
            --fail \
            --show-error \
            --retry 8 \
            --retry-delay 3 \
            --retry-all-errors \
            --connect-timeout 20 \
            --continue-at - \
            --output "$output" \
            "$official_url"
        then
            echo "下载失败：$filename"
            return 1
        fi
    fi

    echo "校验 ZIP：$filename"

    python - "$output" <<'PY'
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])

if not path.exists():
    raise SystemExit(f"文件不存在：{path}")

if path.stat().st_size < 1_000_000:
    raise SystemExit(
        f"文件大小异常：{path.stat().st_size} 字节，"
        "可能下载到的是错误页面或 LFS 指针"
    )

try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"ZIP 中存在损坏文件：{bad}")
except zipfile.BadZipFile:
    raise SystemExit(f"不是有效 ZIP 文件：{path}")

print(
    f"校验通过：{path.name}，"
    f"{path.stat().st_size / 1024 / 1024:.1f} MiB"
)
PY
}

FAILED=0

download_file "AITEX.zip" || FAILED=1
download_file "BTech.zip" || FAILED=1
download_file "MPDD.zip" || FAILED=1

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "至少有一个文件下载失败。"
    echo "已下载的部分文件保留在：$DOWNLOAD_DIR"
    echo "重新运行本脚本即可断点续传。"
    echo
    echo "DNS 检查结果："
    for host in \
        hf-cdn.sufy.com \
        huggingface.co \
        cdn-lfs-us-1.hf.co \
        us.aws.cdn.hf.co
    do
        printf "%-25s " "$host"
        getent hosts "$host" | head -1 || echo "无法解析"
    done
    exit 1
fi

echo
echo "三个压缩包全部下载并校验成功。"
echo "开始解压……"

python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil

target = Path("/home/vina04/ADAR/evaluation/ODinW-13/data")
downloads = target / ".ad_dataset_downloads"

jobs = [
    (downloads / "AITEX.zip", target / "AITEX"),
    (downloads / "BTech.zip", target / "BTAD"),
    (downloads / "MPDD.zip", target / "MPDD"),
]

for archive, destination in jobs:
    if destination.exists():
        count = sum(1 for p in destination.rglob("*") if p.is_file())
        print(
            f"\n{destination.name} 已存在，"
            f"当前有 {count} 个文件，跳过解压。"
        )
        continue

    temporary = target / f".extract_{destination.name}"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    print(f"\n正在解压：{archive.name}")

    with ZipFile(archive) as zf:
        zf.extractall(temporary)

    entries = [
        item
        for item in temporary.iterdir()
        if item.name not in {"__MACOSX", ".DS_Store"}
    ]

    # 压缩包只有一个顶层文件夹时，自动去掉多余层级
    if len(entries) == 1 and entries[0].is_dir():
        entries[0].rename(destination)
        shutil.rmtree(temporary, ignore_errors=True)
    else:
        temporary.rename(destination)

    count = sum(1 for p in destination.rglob("*") if p.is_file())
    print(f"解压完成：{destination.name}，共 {count} 个文件")
PY

echo
echo "================ 最终结果 ================"

for dataset in AITEX ELPV BTAD MPDD; do
    if [ -d "$dataset" ]; then
        count="$(find "$dataset" -type f | wc -l)"
        size="$(du -sh "$dataset" | cut -f1)"
        printf "%-8s 文件数：%-8s 大小：%s\n" \
            "$dataset" "$count" "$size"
    else
        echo "$dataset：不存在"
    fi
done

echo
echo "压缩包暂时保留在：$DOWNLOAD_DIR"
echo "确认数据正常后，可手动删除："
echo "rm -rf '$DOWNLOAD_DIR'"

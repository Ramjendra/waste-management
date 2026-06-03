"""
tools/download_images.py — Download bin images from the internet (no API key needed).

Uses the `icrawler` library which scrapes Bing/Google image search freely.

Run:
    python tools/download_images.py

Optional flags:
    --per-class  30     number of images per class (default 40)
    --engine     bing   search engine: bing | google (default bing — more reliable)

Output:
    data/raw/empty_NNN.jpg
    data/raw/partial_NNN.jpg
    data/raw/full_NNN.jpg
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

RAW_DIR = Path("data/raw")

# Search queries per class — multiple queries give more variety
QUERIES = {
    "empty": [
        "empty trash bin outdoor",
        "empty garbage bin street",
        "empty waste bin clean",
        "empty dustbin outside",
    ],
    "partial": [
        "half full garbage bin",
        "partially filled trash bin",
        "waste bin half full street",
        "dustbin partially filled",
    ],
    "full": [
        "overflowing garbage bin street",
        "full trash bin outdoor",
        "overflowing waste bin",
        "full dustbin overflowing rubbish",
    ],
}


def install_icrawler():
    try:
        import icrawler  # noqa: F401
    except ImportError:
        print("[download] Installing icrawler...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "icrawler",
                               "-q", "--break-system-packages"])
        print("[download] icrawler installed.\n")


def download_class(label: str, queries: list[str], per_class: int, engine: str):
    """Download images for one bin class across multiple search queries."""
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

    class_dir = RAW_DIR / f"_tmp_{label}"
    class_dir.mkdir(parents=True, exist_ok=True)

    per_query = max(1, per_class // len(queries))
    total_saved = 0

    for i, query in enumerate(queries):
        print(f"  [{label}] Query {i+1}/{len(queries)}: '{query}' → {per_query} images")

        tmp_q_dir = class_dir / f"q{i}"
        tmp_q_dir.mkdir(exist_ok=True)

        try:
            if engine == "google":
                crawler = GoogleImageCrawler(
                    storage={"root_dir": str(tmp_q_dir)},
                    log_level=50,   # suppress icrawler logs
                )
            else:
                crawler = BingImageCrawler(
                    storage={"root_dir": str(tmp_q_dir)},
                    log_level=50,
                )

            crawler.crawl(
                keyword=query,
                max_num=per_query,
                file_idx_offset=total_saved,
                filters={"type": "photo"},
            )
        except Exception as e:
            print(f"  [WARNING] Query failed: {e}")
            continue

        # Move downloaded images to data/raw/ with class-prefixed names
        for img_file in sorted(tmp_q_dir.glob("*")):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                dest_name = f"{label}_{total_saved:04d}{img_file.suffix.lower()}"
                dest = RAW_DIR / dest_name
                shutil.move(str(img_file), dest)
                total_saved += 1

        time.sleep(1)   # be polite to search servers

    # Clean up temp dirs
    shutil.rmtree(class_dir, ignore_errors=True)
    print(f"  [{label}] Saved {total_saved} images.\n")
    return total_saved


def verify_downloads():
    """Print a summary of what was downloaded."""
    counts = {}
    for label in QUERIES:
        imgs = list(RAW_DIR.glob(f"{label}_*"))
        counts[label] = len(imgs)

    total = sum(counts.values())
    print("─" * 40)
    print(f"Download summary:")
    for label, count in counts.items():
        bar = "█" * count + "░" * max(0, 40 - count)
        print(f"  {label:8s}  {count:3d} images")
    print(f"  {'TOTAL':8s}  {total:3d} images")
    print("─" * 40)

    if total == 0:
        print("\n[ERROR] No images downloaded.")
        print("        Try switching engine: python tools/download_images.py --engine google")
        return False

    if total < 30:
        print("\n[WARNING] Fewer than 30 images total.")
        print("          Run again with --per-class 60 or try --engine google for more results.")

    print("\nNext step → label the images:")
    print("  python tools/setup_labeling.py\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download bin images from internet")
    parser.add_argument("--per-class", type=int, default=40,
                        help="Images to download per class (default: 40)")
    parser.add_argument("--engine", choices=["bing", "google"], default="bing",
                        help="Search engine (default: bing)")
    args = parser.parse_args()

    install_icrawler()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(RAW_DIR.glob("*.[jJpP][pPnN][gG]")))
    if existing > 0:
        print(f"[download] {existing} images already in {RAW_DIR}/")
        overwrite = input("Download more and add to existing? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    print(f"\n[download] Engine     : {args.engine}")
    print(f"[download] Per class  : {args.per_class}")
    print(f"[download] Output dir : {RAW_DIR.resolve()}")
    print(f"[download] Classes    : {list(QUERIES.keys())}\n")

    for label, queries in QUERIES.items():
        download_class(label, queries, args.per_class, args.engine)

    verify_downloads()


if __name__ == "__main__":
    main()

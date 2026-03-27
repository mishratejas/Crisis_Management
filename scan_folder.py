"""
scan_folder.py
--------------
Drop this in the project root (same level as run_system.py).

Usage:
    python scan_folder.py                        # prompts for folder path
    python scan_folder.py satellite_images       # pass folder directly
    python scan_folder.py satellite_images --pixel-threshold 0.45 --flood-fraction 0.10

How flood is decided (matches vision_agent logic exactly):
    - UNet outputs a per-pixel probability map (H x W, values 0.0-1.0).
    - A pixel is "flooded" if its probability >= PIXEL_THRESHOLD (default 0.45).
      This is the SAME threshold used in vision_agent and route_agent.
    - An image is FLOODED if the fraction of flooded pixels >= FLOOD_FRACTION
      (default 0.10 = 10% of the image area).

    WHY THE PREVIOUS VERSION WAS WRONG:
    Using mean() across the whole map failed because a flooded urban image
    typically has 10-30% of pixels flooded -- not 45%+ of ALL pixels.
    The image you showed had location_07 with mean=0.206 which likely had
    a high flooded-pixel fraction that this version will now catch correctly.
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# Per-pixel threshold: matches vision_agent and route_agent (0.45)
PIXEL_THRESHOLD = 0.45
# Image-level trigger: if this fraction of pixels are flooded -> flood confirmed
# 10% is a safe default for urban satellite imagery
FLOOD_FRACTION = 0.10


class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"

def banner(text, colour=C.CYAN):
    w = 60
    print(f"\n{colour}{C.BOLD}{'─'*w}\n  {text}\n{'─'*w}{C.RESET}")

def info(msg):        print(f"{C.GREY}[INFO ]{C.RESET} {msg}")
def ok(msg):          print(f"{C.GREEN}[ OK  ]{C.RESET} {msg}")
def warn(msg):        print(f"{C.YELLOW}[WARN ]{C.RESET} {msg}")
def err(msg):         print(f"{C.RED}[ERROR]{C.RESET} {msg}")
def flood_alert(msg): print(f"\n{C.RED}{C.BOLD}🚨  {msg}{C.RESET}\n")
def clear_line(msg):  print(f"{C.GREEN}[CLEAR]{C.RESET} {msg}")


# Load model once at startup
banner("Loading UNet flood-segmentation model...", C.CYAN)
try:
    from agents.vision_agent.preprocess         import load_image
    from agents.vision_agent.flood_segmentation import detect_flood
    ok("UNet model loaded (resnet34 encoder, flood_segmentation.py)")
except Exception as e:
    err(f"Could not load vision model: {e}")
    err("Run from project root. Ensure unet_flood_modelN.pth exists.")
    sys.exit(1)


def collect_images(folder: Path) -> list:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def analyse_image(image_path: Path) -> dict:
    """
    Run the real UNet on one image.
    Uses per-pixel thresholding — same logic as vision_agent zones.

    Returns dict:
        prob_map         : H x W float32 numpy array (raw UNet sigmoid output)
        mean_prob        : mean across all pixels (informational only)
        max_prob         : peak pixel probability
        flooded_pixels   : count of pixels >= PIXEL_THRESHOLD
        total_pixels     : total pixels in image
        flooded_fraction : flooded_pixels / total_pixels
        is_flood         : True if flooded_fraction >= FLOOD_FRACTION
    """
    image    = load_image(str(image_path))
    prob_map = detect_flood(image)               # H x W float32

    total   = prob_map.size
    flooded = int((prob_map >= PIXEL_THRESHOLD).sum())
    frac    = flooded / total

    return {
        "prob_map":         prob_map,
        "mean_prob":        float(prob_map.mean()),
        "max_prob":         float(prob_map.max()),
        "flooded_pixels":   flooded,
        "total_pixels":     total,
        "flooded_fraction": frac,
        "is_flood":         frac >= FLOOD_FRACTION,
    }


def _bar(fraction: float, width: int = 20) -> str:
    filled   = int(fraction * width)
    thresh_i = int(FLOOD_FRACTION * width)
    bar = ""
    for i in range(width):
        if i < filled:
            bar += "█" if i < thresh_i else "▓"
        else:
            bar += "░"
    colour = C.RED if fraction >= FLOOD_FRACTION else C.GREEN
    return f"{colour}[{bar}]{C.RESET}"


def run_pipeline(image_path: Path):
    banner("Invoking master_graph crisis pipeline", C.RED)
    try:
        from master_agent.master_graph import master_graph
    except Exception as e:
        err(f"Could not import master_graph: {e}")
        raise

    info(f"master_graph.invoke(satellite_image={image_path.name!r})")
    print()

    master_graph.invoke({
        "satellite_image": str(image_path),
        "field_reports":   [],
        "dispatch_config": {
            "send_sms":       True,
            "generate_audio": True,
            "language":       "English",
        },
    })

    banner("Pipeline complete", C.GREEN)


def scan_folder(folder: Path, pixel_threshold: float, flood_fraction: float):
    global PIXEL_THRESHOLD, FLOOD_FRACTION
    PIXEL_THRESHOLD = pixel_threshold
    FLOOD_FRACTION  = flood_fraction

    images = collect_images(folder)
    if not images:
        warn(f"No image files found in: {folder}")
        return

    banner(f"Satellite folder scan — {len(images)} image(s)", C.CYAN)
    info(f"Folder         : {folder}")
    info(f"Pixel threshold: UNet prob >= {pixel_threshold}  →  pixel flagged flooded")
    info(f"Flood trigger  : flagged pixels >= {flood_fraction*100:.0f}% of image area")
    print()

    flood_found = False

    for idx, img_path in enumerate(images, start=1):
        print(f"{C.WHITE}[{idx:02d}/{len(images):02d}]{C.RESET} {img_path.name}  ", end="", flush=True)

        t0 = time.time()
        try:
            r = analyse_image(img_path)
        except Exception as exc:
            print()
            err(f"Failed on {img_path.name}: {exc}")
            continue

        elapsed = time.time() - t0
        frac    = r["flooded_fraction"]
        stats   = (
            f"flooded={frac*100:5.1f}%  "
            f"max={r['max_prob']:.3f}  "
            f"mean={r['mean_prob']:.3f}  "
            f"{_bar(frac)}  ({elapsed:.1f}s)"
        )

        if r["is_flood"]:
            print(f"{C.RED}{stats}{C.RESET}")
            flood_alert(
                f"FLOOD DETECTED — \"{img_path.name}\"  "
                f"| {frac*100:.1f}% pixels flooded  "
                f"| peak prob={r['max_prob']:.3f}"
            )
            info(f"Flooded pixels : {r['flooded_pixels']:,} / {r['total_pixels']:,}")
            info("Halting folder scan → handing off to master_graph")
            remaining = images[idx:]
            if remaining:
                warn(f"Skipping {len(remaining)} unscanned: "
                     f"{', '.join(p.name for p in remaining)}")
            print()
            flood_found = True
            run_pipeline(img_path)
            break
        else:
            print(f"{C.GREEN}{stats}{C.RESET}")
            clear_line(
                f"\"{img_path.name}\"  "
                f"{frac*100:.1f}% flooded pixels  "
                f"< {flood_fraction*100:.0f}% trigger  →  scanning next"
            )

    if not flood_found:
        print()
        banner("All images scanned — NO FLOOD DETECTED", C.GREEN)
        ok(f"Processed {len(images)} image(s). None hit the {flood_fraction*100:.0f}% pixel trigger.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder of satellite images with the real UNet flood model.\n"
            "Fires the full crisis pipeline on the first flooded image.\n\n"
            "Detection logic:\n"
            "  pixel flagged flooded  →  UNet output >= --pixel-threshold (default 0.45)\n"
            "  image declared flooded →  >= --flood-fraction pixels flagged (default 10%%)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", default=None,
                        help="Satellite image folder. Prompted if omitted.")
    parser.add_argument("--pixel-threshold", type=float, default=PIXEL_THRESHOLD,
                        help=f"Per-pixel UNet cutoff (default {PIXEL_THRESHOLD}). "
                             "Matches vision_agent and route_agent.")
    parser.add_argument("--flood-fraction", type=float, default=FLOOD_FRACTION,
                        help=f"Min fraction of flooded pixels to trigger pipeline "
                             f"(default {FLOOD_FRACTION} = {FLOOD_FRACTION*100:.0f}%%).")
    args = parser.parse_args()

    folder = Path(args.folder).resolve() if args.folder else Path(input("\nEnter satellite image folder path: ").strip()).resolve()

    if not folder.exists() or not folder.is_dir():
        err(f"Invalid folder: {folder}")
        sys.exit(1)

    scan_folder(folder, args.pixel_threshold, args.flood_fraction)


if __name__ == "__main__":
    main()
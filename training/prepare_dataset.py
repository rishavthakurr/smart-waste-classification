"""Inspect a raw image dataset, map labels, split without leakage, verify, and write a report.

Usage:
    python -m training.prepare_dataset --download     # fetch TrashNet from GitHub, then prepare
    python -m training.prepare_dataset                # prepare images already in dataset/raw
    python -m training.prepare_dataset --inspect      # only print the structure; write nothing

Pipeline
 1. INSPECT the real folder structure and refuse to continue if a project class has no data
    (labels are never assumed; the mapping in config.yaml is checked against what is on disk).
 2. Drop corrupt/unsupported files, byte-identical duplicates and identical images with conflicting labels.
 3. Cluster near-duplicates (16x16 thumbnail difference); a cluster is never split across train/validation/test.
 4. Stratified, group-aware 70/15/15 split with a fixed seed (deterministic).
 5. Rebuild dataset/{train,validation,test}/<class>/ from scratch (stale files can never survive).
 6. VERIFY the result (files on disk == manifest, no cluster/md5 crosses splits, every class in every split).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import class_names, load_config, resolve, results_dir, write_json

SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- download / locate
def download_dataset(url: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset.zip"
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:  # network / DNS / HTTP errors
        raise SystemExit(f"Download failed ({exc}). Download manually and extract into {raw_dir}.")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)
    print("Extracted to", raw_dir)


def find_class_root(raw_dir: Path, label_map: dict[str, str]) -> Path:
    """Locate the directory whose sub-folders match the dataset's original labels."""
    cands = [raw_dir] + sorted(p for p in raw_dir.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts)
    for cand in cands:
        subs = {p.name.lower() for p in cand.iterdir() if p.is_dir()}
        if len(subs & set(label_map)) >= 2:
            return cand
    raise SystemExit(f"No class folders matching {sorted(label_map)} found under {raw_dir}.")


# --------------------------------------------------------------------------- 1. inspection
def inspect_raw_dataset(class_root: Path, cfg: dict) -> dict:
    """Describe what is REALLY on disk and check it against config.yaml's label map."""
    label_map, classes = cfg["dataset"]["label_map"], class_names()
    allowed = set(cfg["dataset"]["allowed_extensions"])
    folders = []
    for folder in sorted(p for p in class_root.iterdir() if p.is_dir()):
        files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")]
        folders.append({"folder": folder.name, "files": len(files),
                        "extensions": dict(Counter(f.suffix.lower() for f in files)),
                        "unsupported_files": sum(f.suffix.lower() not in allowed for f in files),
                        "maps_to": label_map.get(folder.name.lower())})
    coverage = {c: sum(f["files"] - f["unsupported_files"] for f in folders if f["maps_to"] == c) for c in classes}
    min_n = cfg["dataset"]["min_images_per_class"]
    problems = [f"project class '{c}' has {n} usable source images (< {min_n}); no dataset folder maps to it "
                f"or it is nearly empty" for c, n in coverage.items() if n < min_n]
    report = {"class_root": str(class_root), "folders": folders, "usable_files_per_class": coverage,
              "unmapped_folders": [f["folder"] for f in folders if f["maps_to"] is None],
              "mapped_but_absent_labels": sorted(set(label_map) - {f["folder"].lower() for f in folders}),
              "problems": problems}
    print(f"\nDataset structure found in {class_root}")
    print(f"{'folder':<14}{'files':>7}  {'maps to':<15}extensions")
    for f in folders:
        print(f"{f['folder']:<14}{f['files']:>7}  {str(f['maps_to'] or 'UNMAPPED (ignored)'):<15}{f['extensions']}")
    if report["mapped_but_absent_labels"]:
        print("Labels in config.yaml with no folder on disk:", report["mapped_but_absent_labels"])
    return report


# --------------------------------------------------------------------------- 2. validation
THUMB = 16


def thumbnail(img: Image.Image) -> np.ndarray:
    """16x16 RGB thumbnail (float32, flattened) used to find near-duplicate photos."""
    return np.asarray(img.convert("RGB").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32).reshape(-1)


def scan_images(class_root: Path, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    allowed = set(cfg["dataset"]["allowed_extensions"])
    label_map, rows, excluded = cfg["dataset"]["label_map"], [], []
    for folder in sorted(p for p in class_root.iterdir() if p.is_dir()):
        orig = folder.name.lower()
        files = sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
        if orig not in label_map:
            excluded += [{"file": str(f), "reason": f"unmapped label '{orig}'"} for f in files]
            continue
        for f in files:
            if f.suffix.lower() not in allowed:
                excluded.append({"file": str(f), "reason": "unsupported extension"}); continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    im.load()
                    w, h = im.size
                    tt = thumbnail(im)
                rows.append({"path": str(f), "original_label": orig, "class": label_map[orig],
                             "md5": hashlib.md5(f.read_bytes()).hexdigest(), "thumb": tt, "height": h, "width": w})
            except Exception as exc:
                excluded.append({"file": str(f), "reason": f"corrupted/unreadable ({type(exc).__name__})"})
    return pd.DataFrame(rows), excluded


def drop_exact_duplicates(df: pd.DataFrame, excluded: list[dict]) -> pd.DataFrame:
    conflict = df.groupby("md5")["class"].nunique()
    conflict_hashes = set(conflict[conflict > 1].index)          # same bytes, different labels
    for _, r in df[df["md5"].isin(conflict_hashes)].iterrows():
        excluded.append({"file": r["path"], "reason": "identical image under conflicting labels"})
    df = df[~df["md5"].isin(conflict_hashes)]
    dup_mask = df.duplicated("md5", keep="first")
    for _, r in df[dup_mask].iterrows():
        excluded.append({"file": r["path"], "reason": "exact duplicate"})
    return df[~dup_mask].reset_index(drop=True)


# --------------------------------------------------------------------------- 3-4. clusters and split
def cluster_near_duplicates(df: pd.DataFrame, max_mad: float) -> tuple[np.ndarray, int]:
    """Union-find over same-class pairs whose thumbnails differ by <= max_mad (mean abs pixel diff).

    Returns (cluster id per row, number of CROSS-class near-duplicate pairs). Cross-class pairs
    are only reported (they hint at label noise); they are not merged.
    """
    n = len(df); parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    feats = np.stack(df["thumb"].to_numpy())
    labels = df["class"].to_numpy()
    cross = 0
    for i in range(n - 1):
        close = np.nonzero(np.abs(feats[i + 1:] - feats[i]).mean(axis=1) <= max_mad)[0] + i + 1
        for j in close:
            if labels[i] == labels[j]:
                parent[find(i)] = find(int(j))
            else:
                cross += 1
    return np.array([find(i) for i in range(n)]), cross


def grouped_stratified_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Assign whole duplicate-clusters to one split, per class, to hit target ratios (deterministic)."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype=object)
    for cls, sub in sorted(df.groupby("class"), key=lambda kv: kv[0]):
        groups = sub.groupby("cluster").size().sort_index()
        order = rng.permutation(groups.index.to_numpy())
        targets = {k: ratios[k] * len(sub) for k in SPLITS}
        filled = {k: 0 for k in SPLITS}
        for g in order:
            k = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
            filled[k] += int(groups[g])
            split.loc[sub.index[sub["cluster"] == g]] = k
    return split


# --------------------------------------------------------------------------- 6. verification
def verify_prepared(manifest: pd.DataFrame, out_dir: Path) -> list[str]:
    """Return a list of problems (empty = split is consistent and leak-free)."""
    problems = []
    on_disk = {str(p.relative_to(out_dir)).replace("\\", "/")
               for s in SPLITS for p in (out_dir / s).rglob("*") if p.is_file()}
    listed = set(manifest["path"])
    if on_disk != listed:
        problems.append(f"files on disk ({len(on_disk)}) differ from manifest ({len(listed)}): "
                        f"{len(on_disk - listed)} unlisted, {len(listed - on_disk)} missing")
    if manifest["path"].duplicated().any():
        problems.append("duplicate paths in manifest")
    if (manifest.groupby("cluster")["split"].nunique() > 1).any():
        problems.append("a near-duplicate cluster spans several splits (leakage)")
    if "md5" in manifest and (manifest.groupby("md5")["split"].nunique() > 1).any():
        problems.append("identical image bytes occur in several splits (leakage)")
    for s in SPLITS:
        missing = set(class_names()) - set(manifest.loc[manifest["split"] == s, "class"])
        if missing:
            problems.append(f"split '{s}' has no images for classes {sorted(missing)}")
    return problems


def make_charts(manifest: pd.DataFrame, out: Path, data_root: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    classes = class_names()
    counts = manifest["class"].value_counts().reindex(classes).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([c.replace("_", " ").title() for c in classes], counts.values, color="#1B5E20")
    ax.set_title("Class distribution (all usable images)"); ax.set_ylabel("Images")
    for i, v in enumerate(counts.values):
        ax.text(i, v, int(v), ha="center", va="bottom")
    plt.xticks(rotation=20); plt.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    pivot = manifest.pivot_table(index="class", columns="split", values="path", aggfunc="count",
                                 fill_value=0).reindex(classes)[list(SPLITS)]
    pivot.plot(kind="bar", stacked=True, figsize=(8, 4.5), color=["#1B5E20", "#81C784", "#2F80ED"])
    plt.title("Images per class and split"); plt.ylabel("Images"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(out / "split_distribution.png", dpi=150); plt.close()

    fig, axes = plt.subplots(len(classes), 4, figsize=(8, 2 * len(classes)))
    for r, cls in enumerate(classes):
        sample = manifest[manifest["class"] == cls].sample(min(4, (manifest["class"] == cls).sum()), random_state=0)
        for c in range(4):
            axes[r, c].axis("off")
            if c < len(sample):
                axes[r, c].imshow(Image.open(data_root / sample.iloc[c]["path"]))
            if c == 0:
                axes[r, c].set_title(cls.replace("_", " ").title(), fontsize=9, loc="left")
    plt.tight_layout(); fig.savefig(out / "sample_images.png", dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download the dataset from the config URL first")
    ap.add_argument("--inspect", action="store_true", help="only inspect the raw dataset; write nothing")
    args = ap.parse_args()
    cfg = load_config(); ds = cfg["dataset"]; seed = cfg["training"]["seed"]
    raw_dir, out_dir = resolve(ds["raw_dir"]), resolve(ds["prepared_dir"])
    if args.download:
        download_dataset(ds["download_url"], raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"{raw_dir} not found. Run with --download or extract the dataset there manually.")

    class_root = find_class_root(raw_dir, ds["label_map"])
    inspection = inspect_raw_dataset(class_root, cfg)
    if inspection["problems"]:
        raise SystemExit("\nDATASET CHECK FAILED:\n  - " + "\n  - ".join(inspection["problems"]) +
                         "\nFix config.yaml (dataset.label_map / classes) or supply the missing data. "
                         "Nothing was written.")
    if args.inspect:
        print("\nInspection passed (nothing written)."); return
    write_json(results_dir() / "dataset_inspection.json", inspection)

    df, excluded = scan_images(class_root, cfg)
    n_scanned = len(df) + len(excluded)
    df = drop_exact_duplicates(df, excluded)
    df["cluster"], cross_pairs = cluster_near_duplicates(df, ds["near_duplicate_thumb_mad"])
    n_near = int((df.groupby("cluster").size() > 1).sum())
    df["split"] = grouped_stratified_split(df, ds["split"], seed)

    for s in SPLITS:                                     # generated artefacts: rebuild from scratch
        shutil.rmtree(out_dir / s, ignore_errors=True)
    rel_paths = []
    for _, r in df.iterrows():
        dst = out_dir / r["split"] / r["class"] / f"{r['original_label']}_{Path(r['path']).name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["path"], dst)
        rel_paths.append(str(dst.relative_to(out_dir)).replace("\\", "/"))
    manifest = df.assign(path=rel_paths)[["path", "class", "original_label", "split", "cluster", "md5", "height", "width"]]
    manifest = manifest.sort_values("path").reset_index(drop=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(excluded, columns=["file", "reason"]).to_csv(results_dir() / "dataset_excluded_images.csv", index=False)

    problems = verify_prepared(manifest, out_dir)
    if problems:
        raise SystemExit("SPLIT VERIFICATION FAILED:\n  - " + "\n  - ".join(problems))

    classes = class_names()
    by_class = manifest["class"].value_counts().reindex(classes).fillna(0).astype(int)
    split_counts = manifest.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(columns=classes, fill_value=0)
    ratio = by_class.max() / max(by_class.min(), 1)
    report = {
        "source": ds["download_url"], "label_map": ds["label_map"], "seed": seed,
        "manifest_sha256": hashlib.sha256((out_dir / "manifest.csv").read_bytes()).hexdigest(),
        "files_scanned": n_scanned, "usable_images": int(len(manifest)), "excluded_images": len(excluded),
        "excluded_by_reason": pd.Series([e["reason"] for e in excluded], dtype=str).value_counts().to_dict(),
        "near_duplicate_clusters_kept_together": n_near,
        "near_duplicate_method": f"16x16 RGB thumbnail mean abs diff <= {ds['near_duplicate_thumb_mad']}",
        "near_duplicate_limitation": "Cannot detect the same physical object photographed from a different pose; test accuracy may be optimistic.",
        "cross_class_near_duplicate_pairs_not_merged": int(cross_pairs),
        "images_per_class": by_class.to_dict(),
        "split_counts": {s: split_counts.loc[s].to_dict() | {"total": int(split_counts.loc[s].sum())} for s in SPLITS},
        "imbalance_ratio_max_over_min": round(float(ratio), 2),
        "imbalance_note": "Class weights are applied during training." if ratio > 1.5 else "Classes roughly balanced.",
        "image_size_range": {"min_height": int(manifest.height.min()), "max_height": int(manifest.height.max()),
                             "min_width": int(manifest.width.min()), "max_width": int(manifest.width.max())},
        "verification": "passed: files on disk == manifest; no cluster or md5 crosses splits; every class in every split",
    }
    write_json(results_dir() / "dataset_report.json", report)
    make_charts(manifest, results_dir() / "dataset", out_dir)
    print(f"\nUsable images: {report['usable_images']}   excluded: {report['excluded_images']}  "
          f"({report['excluded_by_reason']})")
    print("Per class:", report["images_per_class"])
    for s in SPLITS:
        print(f"  {s:<11}", report["split_counts"][s])
    print(f"Imbalance ratio (max/min): {report['imbalance_ratio_max_over_min']}   "
          f"cross-class near-duplicate pairs: {cross_pairs}")
    print("Verification:", report["verification"])
    print(f"Manifest sha256: {report['manifest_sha256'][:16]}...  Report: {results_dir() / 'dataset_report.json'}")


if __name__ == "__main__":
    main()

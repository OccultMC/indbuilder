#!/usr/bin/env python3
"""
VPS Builder Pipeline: Download Features from R2 → Merge → Build FAISS Index → Upload

This worker downloads all feature .npy files and metadata .jsonl files
produced by VPS_Pipeline workers from R2, merges them into a single
features.npy + metadata.json, builds a FAISS IVFPQ index, uploads the
result to R2, and self-destructs.

Progress is logged to stdout in structured format:
    PROGRESS|builder|{step}|{detail}|{pct}|{status}
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from r2_storage import R2Client

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 8448  # MegaLoc feature dimension
WORK_DIR = Path("/app/work")

# FAISS index settings (overridable via env vars)
INDEX_TYPE = os.environ.get("INDEX_TYPE", "ivfpq")
NLIST = int(os.environ.get("NLIST", "1024"))
M = int(os.environ.get("M", "32"))
NBITS = int(os.environ.get("NBITS", "8"))
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "1000000"))
NITER = int(os.environ.get("NITER", "100"))


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

def get_env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        print(f"[FATAL] Missing required env var: {key}")
        sys.exit(1)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Status Reporter
# ═══════════════════════════════════════════════════════════════════════════════

class StatusReporter:
    """Reports progress to R2 and stdout."""

    def __init__(self, r2: R2Client, status_prefix: str):
        self.r2 = r2
        self.status_key = f"Status/{status_prefix}/builder.json"
        self.start_time = time.time()
        self._last_report = 0

    def report(self, step: str, detail: str, pct: int, status: str = "RUNNING"):
        now = time.time()
        elapsed = now - self.start_time

        print(f"PROGRESS|builder|{step}|{detail}|{pct}|{status}")
        sys.stdout.flush()

        # Throttle R2 updates to every 10s
        if now - self._last_report < 10 and status == "RUNNING":
            return

        self._last_report = now
        self.r2.upload_json(self.status_key, {
            "worker": "builder",
            "step": step,
            "detail": detail,
            "pct": pct,
            "status": status,
            "elapsed_seconds": int(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def report_final(self, status: str, detail: str = ""):
        self.report("done", detail, 100, status)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Discover feature files on R2
# ═══════════════════════════════════════════════════════════════════════════════

def discover_feature_files(r2: R2Client, features_prefix: str) -> Tuple[List[str], List[str]]:
    """
    List all .npy and .jsonl files under the features prefix.

    VPS_Pipeline uploads:
      Features/{country}/{state}/{city}/{city}_{worker}.{total}.npy
      Features/{country}/{state}/{city}/Metadata_{city}_{worker}.{total}.jsonl
    """
    print(f"\n{'='*80}")
    print("STEP 1: Discovering feature files on R2")
    print(f"{'='*80}")
    print(f"Prefix: {features_prefix}")

    all_files = r2.list_files(features_prefix)

    npy_files = sorted([f['key'] for f in all_files if f['key'].endswith('.npy')])
    jsonl_files = sorted([f['key'] for f in all_files if f['key'].endswith('.jsonl')])

    print(f"Found {len(npy_files)} feature files (.npy)")
    print(f"Found {len(jsonl_files)} metadata files (.jsonl)")

    for f in npy_files:
        size_mb = next((x['size'] for x in all_files if x['key'] == f), 0) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")

    if not npy_files:
        print("[FATAL] No feature files found!")
        sys.exit(1)
    if not jsonl_files:
        print("[FATAL] No metadata files found!")
        sys.exit(1)

    return npy_files, jsonl_files


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Download all feature files
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_files(r2: R2Client, npy_keys: List[str], jsonl_keys: List[str],
                       reporter: StatusReporter) -> Tuple[List[Path], List[Path]]:
    """Download all feature and metadata files from R2."""
    print(f"\n{'='*80}")
    print("STEP 2: Downloading feature files from R2")
    print(f"{'='*80}")

    download_dir = WORK_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(npy_keys) + len(jsonl_keys)
    downloaded = 0

    npy_paths = []
    for key in npy_keys:
        filename = key.split("/")[-1]
        local_path = download_dir / filename

        if local_path.exists():
            print(f"  [CACHED] {filename}")
        else:
            reporter.report("download", f"Downloading {filename}", int(downloaded / total_files * 100))
            success = r2.download_file(key, str(local_path))
            if not success:
                print(f"  [ERROR] Failed to download {key}")
                sys.exit(1)

        npy_paths.append(local_path)
        downloaded += 1

    jsonl_paths = []
    for key in jsonl_keys:
        filename = key.split("/")[-1]
        local_path = download_dir / filename

        if local_path.exists():
            print(f"  [CACHED] {filename}")
        else:
            reporter.report("download", f"Downloading {filename}", int(downloaded / total_files * 100))
            success = r2.download_file(key, str(local_path))
            if not success:
                print(f"  [ERROR] Failed to download {key}")
                sys.exit(1)

        jsonl_paths.append(local_path)
        downloaded += 1

    reporter.report("download", f"Downloaded {total_files} files", 100)
    return npy_paths, jsonl_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Merge features and metadata
# ═══════════════════════════════════════════════════════════════════════════════

def merge_features_and_metadata(npy_paths: List[Path], jsonl_paths: List[Path],
                                 reporter: StatusReporter) -> Tuple[Path, Path, int]:
    """
    Merge all worker feature files into one features.npy and metadata.json.

    Each worker's .npy is shape (N_worker, 8448).
    Each worker's .jsonl has lines: {"panoid": "...", "lat": ..., "lng": ..., "feature_index": ...}

    The merged metadata.json maps global index → {lat, lng} (same format as build_megaloc_index.py).
    """
    print(f"\n{'='*80}")
    print("STEP 3: Merging features and metadata")
    print(f"{'='*80}")

    # First pass: count total rows
    total_rows = 0
    worker_shapes = []
    for i, npy_path in enumerate(npy_paths):
        mmap = np.load(str(npy_path), mmap_mode='r')
        shape = mmap.shape
        worker_shapes.append(shape)
        total_rows += shape[0]
        print(f"  Worker file {npy_path.name}: {shape[0]:,} vectors")
        del mmap

    print(f"\nTotal vectors to merge: {total_rows:,}")
    total_size_gb = total_rows * FEATURE_DIM * 4 / (1024 ** 3)
    print(f"Merged file size: {total_size_gb:.2f} GB")

    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage(str(WORK_DIR))
    free_gb = disk_usage.free / (1024 ** 3)
    print(f"Disk free: {free_gb:.1f} GB")
    if free_gb < total_size_gb * 1.5:
        print(f"[WARN] Low disk space! Need ~{total_size_gb * 1.5:.1f} GB, have {free_gb:.1f} GB")

    # Create merged memmap
    merged_features_path = WORK_DIR / "features.npy"
    print(f"\nCreating merged memmap: {merged_features_path}")
    reporter.report("merge", "Creating merged feature file", 0)

    merged = np.lib.format.open_memmap(
        str(merged_features_path), mode='w+', dtype='float32',
        shape=(total_rows, FEATURE_DIM)
    )

    # Copy features
    write_offset = 0
    for i, npy_path in enumerate(npy_paths):
        pct = int(i / len(npy_paths) * 80)
        reporter.report("merge", f"Copying {npy_path.name}", pct)

        src = np.load(str(npy_path), mmap_mode='r')
        n_rows = src.shape[0]

        # Copy in chunks to avoid memory issues
        chunk_size = 50000
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            merged[write_offset + start: write_offset + end] = src[start:end]

        write_offset += n_rows
        del src
        gc.collect()

        if (i + 1) % 2 == 0:
            merged.flush()

    merged.flush()
    del merged
    gc.collect()
    print(f"  Merged features written: {total_rows:,} vectors")

    # Merge metadata
    print("\nMerging metadata...")
    reporter.report("merge", "Merging metadata", 85)

    global_metadata = {}
    global_index = 0

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    global_metadata[str(global_index)] = {
                        'lat': float(entry.get('lat', 0)),
                        'lng': float(entry.get('lng', 0)),
                    }
                    global_index += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Still increment to keep alignment with features
                    global_metadata[str(global_index)] = {'lat': 0.0, 'lng': 0.0}
                    global_index += 1

    metadata_path = WORK_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f)

    print(f"  Metadata entries: {len(global_metadata):,}")

    # Validate alignment
    if global_index != total_rows:
        print(f"[WARN] Metadata count ({global_index}) != feature count ({total_rows})")
        print(f"  Using min of both: {min(global_index, total_rows)}")
        total_rows = min(global_index, total_rows)

    reporter.report("merge", f"Merged {total_rows:,} vectors + metadata", 100)
    return merged_features_path, metadata_path, total_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Build FAISS Index
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(features_path: Path, n_vectors: int,
                      reporter: StatusReporter) -> Path:
    """
    Build FAISS IVFPQ index from merged features.

    Uses Inner Product metric (MegaLoc vectors are normalized).
    Reads features via direct file I/O to avoid mmap issues on large datasets.
    """
    import faiss

    print(f"\n{'='*80}")
    print("STEP 4: Building FAISS index")
    print(f"{'='*80}")
    print(f"Vectors: {n_vectors:,}, Dimension: {FEATURE_DIM}")
    print(f"Index type: {INDEX_TYPE}, nlist={NLIST}, m={M}, nbits={NBITS}")

    reporter.report("index", "Preparing index", 0)

    row_bytes = FEATURE_DIM * 4  # float32

    def _read_npy_header(f):
        """Read .npy header and return data offset, compatible with all NumPy versions."""
        f.seek(0)
        version = np.lib.format.read_magic(f)
        # Use public read_array_header_1_0 / 2_0 based on version tuple
        if version[0] == 1:
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        return f.tell()

    def read_features_slice(start_row, end_row):
        """Read a slice of features directly from the .npy file."""
        n_rows = end_row - start_row
        with open(str(features_path), 'rb') as f:
            data_offset = _read_npy_header(f)
            f.seek(data_offset + start_row * row_bytes)
            raw = f.read(n_rows * row_bytes)
            result = np.frombuffer(raw, dtype=np.float32).reshape(n_rows, FEATURE_DIM).copy()
        return result

    def read_features_by_indices(indices):
        """Read specific rows by index from the .npy file."""
        n_rows = len(indices)
        result = np.empty((n_rows, FEATURE_DIM), dtype=np.float32)
        with open(str(features_path), 'rb') as f:
            data_offset = _read_npy_header(f)
            for i, idx in enumerate(indices):
                f.seek(data_offset + int(idx) * row_bytes)
                raw = f.read(row_bytes)
                result[i] = np.frombuffer(raw, dtype=np.float32)
        return result

    # Auto-adjust nlist if needed
    nlist = NLIST
    max_nlist = n_vectors // 39
    if nlist > max_nlist:
        nlist = max(64, 2 ** int(np.log2(max_nlist)))
        print(f"  Auto-adjusted nlist: {NLIST} -> {nlist}")

    # Create index with Inner Product metric
    metric = faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexFlatIP(FEATURE_DIM)
    index = faiss.IndexIVFPQ(quantizer, FEATURE_DIM, nlist, M, NBITS, metric)

    # Set clustering iterations (default FAISS is 25, we use 100 for better quality)
    index.cp.niter = NITER
    print(f"  Clustering iterations (niter): {NITER}")

    # Training
    train_samples = min(n_vectors, TRAIN_SAMPLES)
    print(f"\nSampling {train_samples:,} vectors for training...")
    reporter.report("index", f"Sampling {train_samples:,} training vectors", 5)

    rng = np.random.default_rng(42)
    train_indices = np.sort(rng.choice(n_vectors, size=train_samples, replace=False))

    print("Reading training vectors from disk...")
    reporter.report("index", "Reading training data", 10)
    train_data = read_features_by_indices(train_indices)
    faiss.normalize_L2(train_data)
    print(f"Training data loaded: {train_data.nbytes / (1024 ** 2):.0f} MB")

    print("Training index...")
    reporter.report("index", "Training FAISS index", 15)
    index.train(train_data)
    del train_data
    gc.collect()
    print("Training complete.")

    # Adding vectors
    add_batch_size = 10_000
    total_batches = (n_vectors + add_batch_size - 1) // add_batch_size
    print(f"\nAdding {n_vectors:,} vectors in batches of {add_batch_size:,}...")
    reporter.report("index", "Adding vectors", 20)

    for batch_idx, start in enumerate(range(0, n_vectors, add_batch_size)):
        end = min(start + add_batch_size, n_vectors)
        batch = read_features_slice(start, end)
        faiss.normalize_L2(batch)
        index.add(batch)
        del batch

        if (batch_idx + 1) % 100 == 0:
            pct = 20 + int(batch_idx / total_batches * 70)
            reporter.report("index", f"Added {end:,}/{n_vectors:,} vectors", pct)
            gc.collect()

    print(f"Index built with {index.ntotal} vectors")

    # Save index
    index_path = WORK_DIR / "megaloc.index"
    print(f"\nSaving index to {index_path}...")
    reporter.report("index", "Saving index file", 92)
    faiss.write_index(index, str(index_path))

    index_size_mb = os.path.getsize(str(index_path)) / (1024 * 1024)
    raw_size_mb = n_vectors * FEATURE_DIM * 4 / (1024 * 1024)
    print(f"  Index size: {index_size_mb:.2f} MB")
    print(f"  Raw features size: {raw_size_mb:.2f} MB")
    print(f"  Compression ratio: {raw_size_mb / index_size_mb:.1f}x")

    # Save config
    config = {
        'n_vectors': n_vectors,
        'dimension': FEATURE_DIM,
        'index_type': INDEX_TYPE,
        'nlist': nlist,
        'm': M,
        'nbits': NBITS,
        'index_file': 'megaloc.index',
        'metadata_file': 'metadata.json',
        'features_file': 'features.npy',
    }
    config_path = WORK_DIR / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    reporter.report("index", f"Index built: {index.ntotal:,} vectors, {index_size_mb:.0f} MB", 95)
    return index_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Upload results to R2
# ═══════════════════════════════════════════════════════════════════════════════

def upload_results(r2: R2Client, features_prefix: str, reporter: StatusReporter):
    """Upload megaloc.index, metadata.json, and config.json to R2."""
    print(f"\n{'='*80}")
    print("STEP 5: Uploading results to R2")
    print(f"{'='*80}")

    upload_prefix = f"Index/{'/'.join(features_prefix.rstrip('/').split('/')[1:])}"

    files_to_upload = [
        (WORK_DIR / "megaloc.index", f"{upload_prefix}/megaloc.index"),
        (WORK_DIR / "metadata.json", f"{upload_prefix}/metadata.json"),
        (WORK_DIR / "config.json", f"{upload_prefix}/config.json"),
    ]

    for local_path, r2_key in files_to_upload:
        if not local_path.exists():
            print(f"  [SKIP] {local_path.name} not found")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_path.name} ({size_mb:.1f} MB) -> {r2_key}")
        reporter.report("upload", f"Uploading {local_path.name}", 95)

        # Retry indefinitely for the index file (it's precious)
        max_attempts = 10 if "index" in local_path.name else 3
        for attempt in range(1, max_attempts + 1):
            success = r2.upload_file(str(local_path), r2_key)
            if success:
                break
            print(f"  [RETRY] Upload failed, attempt {attempt}/{max_attempts}")
            time.sleep(min(60, 2 ** attempt))
        else:
            print(f"  [ERROR] Failed to upload {local_path.name} after {max_attempts} attempts")

    reporter.report("upload", "All files uploaded", 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Upload logs & Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_instance_id(r2: R2Client, features_prefix: str) -> str:
    """Detect our instance ID from R2 or env."""
    instance_id = os.environ.get("INSTANCE_ID", "")
    if instance_id:
        return instance_id

    # Try R2 lookup
    try:
        data = r2.download_json(f"Status/{features_prefix}/builder_instance.json")
        if data and 'instance_id' in data:
            return str(data['instance_id'])
    except Exception:
        pass

    # Fallback: vastai CLI
    api_key = os.environ.get("VAST_API_KEY", "")
    if api_key:
        try:
            result = subprocess.run(
                ["vastai", "--api-key", api_key, "show", "instances", "--raw"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                instances = json.loads(result.stdout)
                if len(instances) == 1:
                    return str(instances[0].get("id", ""))
        except Exception:
            pass

    return ""


def upload_logs(r2: R2Client, features_prefix: str, log_path: Path):
    """Upload the full instance log to R2."""
    if not log_path.exists():
        print("[WARN] No log file to upload")
        return

    r2_key = f"Logs/{'/'.join(features_prefix.rstrip('/').split('/')[1:])}/builder.log"
    size_mb = log_path.stat().st_size / (1024 * 1024)
    print(f"Uploading instance log ({size_mb:.2f} MB) -> {r2_key}")

    for attempt in range(3):
        success = r2.upload_file(str(log_path), r2_key)
        if success:
            print(f"Log uploaded to {r2_key}")
            return
        print(f"[RETRY] Log upload attempt {attempt+1} failed")
        time.sleep(2 ** attempt)

    print("[WARN] Failed to upload log after 3 attempts")


def self_destruct(r2: R2Client, features_prefix: str):
    """Destroy this instance via vastai CLI."""
    api_key = os.environ.get("VAST_API_KEY", "")
    if not api_key:
        print("[WARN] No VAST_API_KEY — cannot self-destruct")
        return

    instance_id = _detect_instance_id(r2, features_prefix)
    if not instance_id:
        print("[WARN] Could not detect instance ID — cannot self-destruct")
        return

    print(f"\n[SELF-DESTRUCT] Destroying instance {instance_id}...")
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["vastai", "--api-key", api_key, "destroy", "instance", instance_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"[SELF-DESTRUCT] Instance {instance_id} destroyed.")
                return
            print(f"[SELF-DESTRUCT] Attempt {attempt+1} failed: {result.stderr}")
        except Exception as e:
            print(f"[SELF-DESTRUCT] Attempt {attempt+1} error: {e}")
        time.sleep(2 ** attempt)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

class _LogTee:
    """Tee stdout/stderr to a log file while preserving console output."""

    def __init__(self, log_path: Path):
        self.log_file = open(str(log_path), 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def start(self):
        sys.stdout = self._TeeStream(self.stdout, self.log_file)
        sys.stderr = self._TeeStream(self.stderr, self.log_file)

    def stop(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    class _TeeStream:
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, data):
            self.original.write(data)
            try:
                self.log_file.write(data)
                self.log_file.flush()
            except Exception:
                pass

        def flush(self):
            self.original.flush()
            try:
                self.log_file.flush()
            except Exception:
                pass


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log_path = WORK_DIR / "builder.log"
    tee = _LogTee(log_path)
    tee.start()

    print("=" * 80)
    print("  VPS BUILDER PIPELINE — Feature Merge + FAISS Index")
    print("=" * 80)

    # Environment
    features_prefix = get_env("FEATURES_BUCKET_PREFIX")
    city_name = get_env("CITY_NAME")

    print(f"Features prefix: {features_prefix}")
    print(f"City: {city_name}")
    print(f"Work dir: {WORK_DIR}")

    # Init R2
    r2 = R2Client()
    reporter = StatusReporter(r2, features_prefix)
    reporter.report("init", "Starting builder pipeline", 0)

    try:
        # Step 1: Discover files
        npy_keys, jsonl_keys = discover_feature_files(r2, features_prefix)

        # Step 2: Download
        npy_paths, jsonl_paths = download_all_files(r2, npy_keys, jsonl_keys, reporter)

        # Step 3: Merge
        features_path, metadata_path, total_vectors = merge_features_and_metadata(
            npy_paths, jsonl_paths, reporter
        )

        # Step 4: Build FAISS index
        index_path = build_faiss_index(features_path, total_vectors, reporter)

        # Step 5: Upload
        upload_results(r2, features_prefix, reporter)

        reporter.report_final("COMPLETED", f"Index built with {total_vectors:,} vectors")
        print("\n" + "=" * 80)
        print("  DONE! Index built and uploaded successfully.")
        print("=" * 80)

    except Exception as e:
        reporter.report_final("FAILED", str(e))
        print(f"\n[FATAL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

        # Upload log before pausing
        tee.stop()
        upload_logs(r2, features_prefix, log_path)

        print("[INFO] Container kept alive for debugging. SSH in to investigate.")
        sys.stdout.flush()
        import signal
        signal.pause()
        return

    # Upload full instance log
    tee.stop()
    upload_logs(r2, features_prefix, log_path)

    # Step 6: Self-destruct
    self_destruct(r2, features_prefix)


if __name__ == "__main__":
    main()

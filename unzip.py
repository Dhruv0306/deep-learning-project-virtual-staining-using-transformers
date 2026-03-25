"""
Dataset extraction utility.

Extracts the E-Staining DermaRepo ZIP archive into a flat directory structure
suitable for the preprocessing pipeline.  Uses streaming extraction with a
10 MB copy buffer so that very large whole-slide images are handled without
loading them fully into memory.

Usage::

    # Edit ZIP_PATH and OUTPUT_DIR below, then run:
    python unzip.py
"""

import os
import zipfile
import shutil

# Path to the source ZIP file containing the dataset archive.
ZIP_PATH = os.path.join("data", "data", "E-Staining DermaRepo.zip")
# Directory where extracted files will be written.
OUTPUT_DIR = os.path.join("data", "E-Staining")


def extract_zip_streaming(zip_path, out_dir, buffer_bytes=1024 * 1024 * 10):
    """
    Extract a ZIP archive with bounded memory usage.

    Args:
        zip_path (str): Path to the source ZIP file.
        out_dir (str): Destination directory.
        buffer_bytes (int): Streaming buffer size used by copyfileobj().

    Returns:
        None: Files are extracted into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target_path = os.path.join(out_dir, info.filename)

            if info.is_dir():
                os.makedirs(target_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Stream each member to avoid loading large pathology files in memory.
            with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=buffer_bytes)


def main():
    """Run extraction using configured ZIP_PATH and OUTPUT_DIR values."""
    extract_zip_streaming(ZIP_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    main()

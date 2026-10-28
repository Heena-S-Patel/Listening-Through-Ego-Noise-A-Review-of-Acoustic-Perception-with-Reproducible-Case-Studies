"""
Extract only the robot-head (benchmark2) recordings of chosen LOCATA tasks from the 6.2 GB Zenodo dev.zip
without downloading the whole archive.

    1. curl the last 10 MB of dev.zip            -> tail.bin   (holds the zip's central directory)
    2. python tools/locata_partial.py plan        -> ranges.txt (byte ranges of the wanted members)
    3. curl each range                            -> parts/<start>.bin
    4. python tools/locata_partial.py extract     -> data/LOCATA/dev/taskN/recordingK/benchmark2/...

Steps 1 and 3 are plain HTTP range requests (see README); this script does no network access.
"""
import io
import os
import sys
import zipfile

ZIP_SIZE = 6207354195
TAIL = 10_000_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("LOCATA_WORK", os.path.join(ROOT, "data", "LOCATA_download"))
OUT = os.environ.get("LOCATA_OUT", os.path.join(ROOT, "data", "LOCATA"))
TASKS = ("task1", "task3", "task5")
ARRAY = "benchmark2"
MARGIN = 4096            # local header (30 bytes + name + extra field) fits in this


class RangeFile(io.RawIOBase):
    """Read-only file made of byte ranges; reading outside them raises, so a missing part is noticed."""

    def __init__(self, size, chunks):
        self.size, self.chunks, self.pos = size, sorted(chunks), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for start, data in self.chunks:
            if start <= self.pos < start + len(data):
                k = min(n, start + len(data) - self.pos)
                b[:k] = data[self.pos - start: self.pos - start + k]
                self.pos += k
                return k
        raise IOError(f"byte {self.pos} not downloaded")


def wanted(name):
    parts = name.replace("\\", "/").split("/")
    return ARRAY in parts and any(t in parts for t in TASKS) and not name.endswith("/")


def tail_chunk():
    with open(os.path.join(WORK, "tail.bin"), "rb") as f:
        return (ZIP_SIZE - TAIL, f.read())


def plan():
    zf = zipfile.ZipFile(RangeFile(ZIP_SIZE, [tail_chunk()]))
    infos = [i for i in zf.infolist() if wanted(i.filename)]
    total = 0
    with open(os.path.join(WORK, "ranges.txt"), "w") as f:
        for i in infos:
            a = i.header_offset
            b = min(ZIP_SIZE - 1, a + MARGIN + i.compress_size)
            f.write(f"{a} {b}\n")
            total += b - a + 1
    print(f"{len(zf.infolist())} members in archive, {len(infos)} wanted, {total / 1e6:.0f} MB to fetch")
    for i in infos[:12]:
        print("  ", i.filename, i.file_size)


def extract():
    chunks = [tail_chunk()]
    parts = os.path.join(WORK, "parts")
    for fn in os.listdir(parts):
        with open(os.path.join(parts, fn), "rb") as f:
            chunks.append((int(fn.split(".")[0]), f.read()))
    zf = zipfile.ZipFile(RangeFile(ZIP_SIZE, chunks))
    n = 0
    for i in zf.infolist():
        if wanted(i.filename):
            zf.extract(i, OUT)
            n += 1
    print(f"extracted {n} files to {OUT}")


if __name__ == "__main__":
    {"plan": plan, "extract": extract}[sys.argv[1]]()

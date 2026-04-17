"""
Dual-output logging: every print() goes to both the terminal and a log file.
tqdm writes to stderr so it stays on the terminal only — logs stay clean.
"""
import io
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime


# Strip ANSI escape codes (colour, cursor movement) before writing to file
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mABCDEFGHJKSTfhilmnsu]|\r")


class _Tee(io.TextIOBase):
    """Writes to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str, original_stdout):
        self._stdout = original_stdout
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self._stdout.write(text)
        self._file.write(_ANSI_RE.sub("", text))
        return len(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Needed so tqdm and other libs can check isatty on the underlying stream
    def isatty(self) -> bool:
        return self._stdout.isatty()

    @property
    def encoding(self):
        return self._stdout.encoding


@contextmanager
def run_log(out_dir: str, filename: str = "train.log"):
    """
    Context manager that tees all print() output to out_dir/train.log.
    Usage:
        with run_log(out_dir):
            ... training code ...
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)

    tee = _Tee(log_path, sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        print(f"[log] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  →  {log_path}")
        yield log_path
    finally:
        sys.stdout = original_stdout
        tee.close()

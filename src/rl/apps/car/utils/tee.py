import contextlib
import io
import sys


class _StdoutCapture:
    def __init__(self):
        self.stdout = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message: str):
        self.stdout.write(message)
        self.buffer.write(message)

    def flush(self):
        self.stdout.flush()
        self.buffer.flush()

    def get(self) -> str:
        return self.buffer.getvalue()

    def __enter__(self) -> "_StdoutCapture":
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout


@contextlib.contextmanager
def capture_stdout():
    with _StdoutCapture() as output:
        yield output

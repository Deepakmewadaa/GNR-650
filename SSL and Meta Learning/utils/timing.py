import time


def fmt_secs(s):
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0

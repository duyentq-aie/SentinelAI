"""InferenceWorker: one inference per frame for many clients, idle when unwatched, no hot loops."""
import threading
import time
from types import SimpleNamespace

import numpy as np

from apps.web.worker import InferenceWorker


class FakePipeline:
    def __init__(self, delay=0.0):
        self.processed, self.resets, self.delay = 0, 0, delay

    def process(self, frame):
        time.sleep(self.delay)
        self.processed += 1
        return SimpleNamespace(frame=frame)

    def reset(self):
        self.resets += 1


class FakeSource:
    """n frames, then read() fails (a finite video file) unless `live`."""
    instances = []

    def __init__(self, spec, n=5, openable=True, live=False):
        self.spec, self.n, self.i, self.openable = spec, n, 0, openable
        self.is_file = not live
        self.reads, self.rewinds, self.released = 0, 0, False
        FakeSource.instances.append(self)

    def is_open(self):
        return self.openable and not self.released

    def read(self):
        self.reads += 1
        if self.i < self.n:
            self.i += 1
            return True, np.full((4, 4, 3), self.i, np.uint8)
        return False, None

    def rewind(self):
        self.i = 0
        self.rewinds += 1

    def release(self):
        self.released = True


def make(factory=FakeSource, delay=0.0, **kw):
    FakeSource.instances = []
    pipe = FakePipeline(delay)
    w = InferenceWorker(pipe, "video.mp4", source_factory=factory, encode=lambda f: bytes([int(f[0, 0, 0])]),
                        read_retry=0.01, open_retry=0.01, **kw)
    return w, pipe


def test_idle_worker_reads_nothing():
    w, pipe = make()
    time.sleep(0.2)
    assert pipe.processed == 0 and FakeSource.instances == []      # not even started


def test_many_clients_share_one_inference_per_frame():
    w, pipe = make(delay=0.005)
    got = {}

    def client(name):
        last, ids = 0, []
        while len(ids) < 8:
            item = w.next_jpeg(last, timeout=2)
            assert item is not None
            last = item[0]
            ids.append(last)
        got[name] = ids

    ts = [threading.Thread(target=client, args=(k,)) for k in "abc"]
    [t.start() for t in ts]; [t.join(5) for t in ts]
    w.stop()
    assert set(got) == {"a", "b", "c"}
    assert all(v == sorted(v) and len(set(v)) == 8 for v in got.values())
    # exactly one process() per published frame, however many clients are connected
    assert pipe.processed == w.frames_processed == w._frame_id
    assert pipe.processed < 8 * 3                            # per-client pipelines would need >= 24 inferences


def test_jpeg_is_encoded_once_per_frame():
    calls = []
    w, _ = make()
    w._encode = lambda f: calls.append(1) or b"x"
    a = w.next_jpeg(0); b = w.next_jpeg(a[0] - 1)           # second client asks for the same frame id
    w.stop()
    assert a[0] == b[0] or b[0] > a[0]
    assert len(calls) <= 2                                   # same frame id never encoded twice


def test_file_loop_resets_pipeline_state():
    w, pipe = make()
    last = 0
    while w.loops < 2:                                       # 5-frame file loops repeatedly
        item = w.next_jpeg(last, timeout=2)
        assert item is not None
        last = item[0]
    w.stop()
    assert pipe.resets >= 1 + 2                              # once at open + once per loop
    assert FakeSource.instances[0].rewinds >= 2


def test_changing_source_resets_state_and_reopens():
    w, pipe = make()
    w.next_jpeg(0)
    resets_before = pipe.resets
    w.set_source("other.mp4")
    last = 0
    deadline = time.time() + 3
    while len(FakeSource.instances) < 2 and time.time() < deadline:
        w.next_jpeg(last, timeout=0.2)
    w.stop()
    assert len(FakeSource.instances) == 2 and FakeSource.instances[0].released
    assert FakeSource.instances[1].spec == "other.mp4"
    assert pipe.resets > resets_before


def test_unopenable_source_does_not_spin():
    w, pipe = make(factory=lambda spec: FakeSource(spec, openable=False))
    w.next_jpeg(0, timeout=0.3)                              # starts the thread, no frame will come
    time.sleep(0.3)
    w.stop()
    assert pipe.processed == 0
    assert len(FakeSource.instances) < 100                   # retries are paced by open_retry, not a busy loop


def test_worker_idles_after_clients_stop():
    w, pipe = make(idle_after=0.2)
    w.next_jpeg(0)
    time.sleep(0.5)                                          # nobody polling -> worker goes idle
    n = pipe.processed
    time.sleep(0.3)
    w.stop()
    assert pipe.processed == n                               # no inference while unwatched
    assert w.status()["running"] is False


def test_nothing_is_encoded_without_a_client_and_the_default_encoder_uses_the_configured_quality(monkeypatch):
    import cv2
    seen = []
    real = cv2.imencode
    monkeypatch.setattr(cv2, "imencode", lambda ext, img, params=None: (seen.append(params), real(ext, img, params or []))[1])
    FakeSource.instances = []
    pipe = FakePipeline()
    w = InferenceWorker(pipe, "video.mp4", source_factory=FakeSource, jpeg_quality=63, idle_after=0.2, read_retry=0.01, open_retry=0.01)
    time.sleep(0.3)
    assert seen == [] and pipe.processed == 0               # no client: not even read, let alone encoded
    w.next_jpeg(0)                                          # a client appears
    time.sleep(0.5)                                         # worker keeps producing frames for ~idle_after, but nobody asks for JPEGs
    n_encoded_after_one_client_request = len(seen)
    w.stop()
    assert n_encoded_after_one_client_request == 1          # frames are encoded lazily, only when a client asks for one
    assert seen[0] == [cv2.IMWRITE_JPEG_QUALITY, 63]


def test_status_reports_rolling_fps_latency_and_backend():
    w, pipe = make(delay=0.01, backend={"runtime": "pytorch", "device": "cpu"})
    assert w.status()["fps"] is None and w.status()["latency_ms"] is None
    last = 0
    for _ in range(12):
        last = w.next_jpeg(last, timeout=2)[0]
    st = w.status()
    w.stop()
    assert st["backend"] == {"runtime": "pytorch", "device": "cpu"} and st["jpeg_quality"] == 80
    assert 5 < st["fps"] < 120 and 8 <= st["latency_ms"] < 100              # ~10 ms of fake processing per frame

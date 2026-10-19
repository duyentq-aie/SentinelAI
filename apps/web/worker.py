"""One inference worker shared by all HTTP clients.

The old code ran a full pipeline (and read from the shared capture) once per connected client.
Here a single thread owns the source and the Pipeline, publishes each annotated frame once,
and clients only wait for the next frame and get its JPEG (encoded at most once per frame, and
only while somebody is polling). With no client polling for `idle_after` seconds the worker
stops reading/inferring.
"""
import threading
import time
from collections import deque

import cv2

from sentinel.source import VideoSource


class InferenceWorker:
    def __init__(self, pipeline, source, source_factory=VideoSource, idle_after=5.0,
                 open_retry=1.0, read_retry=0.05, reopen_after_failures=100, encode=None, jpeg_quality=90, clock=time.monotonic, backend=None):
        self.pipeline = pipeline
        self._factory = source_factory
        self._pending = source              # source spec waiting to be (re)opened by the worker thread
        self._source = None                 # current source spec (for status)
        self.idle_after, self.open_retry, self.read_retry = idle_after, open_retry, read_retry
        self.reopen_after_failures = reopen_after_failures
        self.jpeg_quality = jpeg_quality
        self._encode = encode or (lambda f: cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1].tobytes())
        self._clock = clock
        self._lock = threading.Lock()       # guards _pending / _last_poll / thread start
        self._cond = threading.Condition()  # guards published frame
        self._enc_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._last_poll = -1e18
        self._frame_id, self._frame = 0, None
        self._jpeg_id, self._jpeg = -1, None
        self.frames_processed = 0
        self.loops = 0
        self.backend = backend or {}
        self._timing = deque(maxlen=60)       # (finished_at, seconds spent in pipeline.process)

    # ---- control
    def set_source(self, source):
        with self._lock:
            self._pending = source
        return str(source)

    def status(self):
        t = list(self._timing)
        fps = (len(t) - 1) / (t[-1][0] - t[0][0]) if len(t) >= 2 and t[-1][0] > t[0][0] else None
        return {"running": self._thread is not None and self._thread.is_alive() and self._active(),
                "source": str(self._source if self._source is not None else self._pending),
                "frames_processed": self.frames_processed, "loops": self.loops,
                "fps": round(fps, 1) if fps else None,                                   # processing rate over the last <= 60 frames
                "latency_ms": round(1000 * sum(d for _, d in t) / len(t), 1) if t else None,   # mean time inside pipeline.process
                "backend": self.backend, "jpeg_quality": self.jpeg_quality,
                "red_light": self.pipeline.red_light_status() if hasattr(self.pipeline, "red_light_status") else None,   # state active | inactive | disabled | legacy_zone (+ reasons)
                "run": getattr(getattr(self.pipeline, "recorder", None), "run", None)}      # evidence run number = pass over the video since the source was opened

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---- client side
    def next_jpeg(self, last_id=0, timeout=2.0):
        """Block until a frame newer than `last_id` exists. -> (frame_id, jpeg_bytes) or None on timeout."""
        with self._lock:
            self._last_poll = self._clock()
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name="inference-worker", daemon=True)
                self._thread.start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame_id > last_id, timeout=timeout):
                return None
            fid, frame = self._frame_id, self._frame
        with self._enc_lock:
            if self._jpeg_id != fid:
                self._jpeg, self._jpeg_id = self._encode(frame), fid
            return self._jpeg_id, self._jpeg

    # ---- worker thread
    def _active(self):
        return self._clock() - self._last_poll <= self.idle_after

    def _open(self, spec):
        src = self._factory(spec)
        self._source = spec
        fps = getattr(src, "fps", None)                     # durations in the config follow the frame rate of this source
        if fps and hasattr(self.pipeline, "set_fps"):
            self.pipeline.set_fps(fps)
        return src

    def _run(self):
        src, failures = None, 0
        while not self._stop.is_set():
            if not self._active():                          # nobody watching: no reads, no inference
                if src is not None and not src.is_file:
                    src.release(); src = None               # do not hold a live camera while idle
                self._stop.wait(0.1)
                continue
            with self._lock:
                spec, self._pending = self._pending, None
            if spec is not None:                            # new source requested
                if src is not None:
                    src.release()
                src = self._open(spec)
                self.pipeline.reset()
                failures = 0
            if src is None:
                with self._lock:
                    spec = self._source
                if spec is None:
                    self._stop.wait(self.open_retry); continue
                src = self._open(spec)
            if not src.is_open():                           # cannot open: wait instead of spinning
                self._stop.wait(self.open_retry)
                src.release(); src = self._open(self._source)
                continue
            ok, frame = src.read()
            if not ok:
                failures += 1
                if src.is_file:
                    src.rewind(); self.pipeline.reset(); self.loops += 1   # ids restart on every loop
                    if failures > 1:                                      # empty/unreadable file: avoid a hot loop
                        self._stop.wait(self.read_retry)
                else:
                    self._stop.wait(self.read_retry)
                    if failures >= self.reopen_after_failures:            # dropped stream: reconnect
                        src.release(); src = self._open(self._source); failures = 0
                continue
            failures = 0
            t0 = time.perf_counter()
            result = self.pipeline.process(frame)
            self._timing.append((time.perf_counter(), time.perf_counter() - t0))
            self.frames_processed += 1
            with self._cond:
                self._frame, self._frame_id = result.frame, self._frame_id + 1
                self._cond.notify_all()
        if src is not None:
            src.release()

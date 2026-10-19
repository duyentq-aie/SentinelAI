"""Video source wrapper (file / camera index / network stream)."""
import cv2


def is_live_source(source):
    """Camera indexes and network streams (http/rtsp/...) are live; plain paths are files that can loop."""
    return isinstance(source, int) or (isinstance(source, str) and "://" in source)


class VideoSource:
    def __init__(self, source, opener=cv2.VideoCapture):
        self.source = int(source) if isinstance(source, str) and source.strip().isdigit() else source
        self.is_file = not is_live_source(self.source)
        self._cap = opener(self.source)

    @property
    def fps(self):
        """Nominal frame rate reported by the capture, or None when unknown (many live streams report 0)."""
        v = self._cap.get(cv2.CAP_PROP_FPS) if self._cap is not None else 0
        return float(v) if v and 1.0 <= v <= 240.0 else None

    def is_open(self):
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        """-> (ok, frame)."""
        return self._cap.read()

    def rewind(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

from collections import deque

import cv2
import numpy as np

from Motor_y_Captura_v2 import VideoEngineThreaded


class SilhouetteTrailEffect:
    def __init__(self, trail_length=12, decay=0.78):
        self.trail_length = trail_length
        self.decay = decay
        self.masks = deque(maxlen=trail_length)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.background = self._create_background_model()
        self.has_background_frame = False

    @staticmethod
    def _create_background_model():
        return cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=32,
            detectShadows=False,
        )

    def reset(self):
        self.background = self._create_background_model()
        self.masks.clear()
        self.has_background_frame = False

    def apply(self, frame):
        foreground = self.background.apply(frame)
        if not self.has_background_frame:
            self.has_background_frame = True
            foreground = np.zeros_like(foreground)

        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, self.kernel
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, self.kernel
        )
        foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
        self.masks.append(foreground)

        trail_alpha = np.zeros(foreground.shape, dtype=np.float32)
        for age, mask in enumerate(reversed(self.masks)):
            weighted_mask = mask.astype(np.float32) * (
                self.decay**age / 255.0
            )
            np.maximum(trail_alpha, weighted_mask, out=trail_alpha)

        current_alpha = foreground.astype(np.float32) / 255.0
        trail_color = np.array((255, 170, 35), dtype=np.float32)
        color = trail_alpha[..., None] * trail_color
        color = color * (1.0 - current_alpha[..., None]) + (
            255.0 * current_alpha[..., None]
        )

        alpha = (trail_alpha * 255).astype(np.uint8)
        return np.dstack((color.astype(np.uint8), alpha))


def run_effect(engine):
    effect = SilhouetteTrailEffect()
    last_processed_frame = None
    processed = None

    cv2.namedWindow("Captura - Motor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Silueta con estela", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                frame = frames[-1]
                if frame is not last_processed_frame:
                    processed = effect.apply(frame)
                    last_processed_frame = frame

                cv2.imshow("Captura - Motor", frame)
                if processed is not None:
                    cv2.imshow("Silueta con estela", processed[:, :, :3])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                effect.reset()
                last_processed_frame = None
                processed = None
    finally:
        cv2.destroyAllWindows()


def main():
    engine = VideoEngineThreaded(
        src=0, width=640, height=480, buffer_size=30
    ).start()
    try:
        run_effect(engine)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

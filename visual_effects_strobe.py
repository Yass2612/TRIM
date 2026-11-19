from collections import deque
import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteStrobeFreezeEffect:
    """
    Efecto Strobe Freeze (ecos cronofotográficos):
    Congela capturas de la silueta cada cierto intervalo de fotogramas
    y las superpone sobre el vídeo en tiempo real con un tinte configurable.
    """

    def __init__(
        self,
        capture_interval=10,
        max_echoes=5,
        echo_opacity=0.65,
        tint_bgr=(255, 200, 80),
        line_thickness=2,
    ):
        self.capture_interval = capture_interval
        self.max_echoes = max_echoes
        self.echo_opacity = echo_opacity
        self.tint_bgr = np.array(tint_bgr, dtype=np.float32) if tint_bgr is not None else None
        self.line_thickness = line_thickness

        self.echoes = deque(maxlen=max_echoes)
        self.frame_count = 0
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()
        self.echoes.clear()
        self.frame_count = 0

    @staticmethod
    def _as_uint8_mask(mask):
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def apply(self, frame):
        h, w = frame.shape[:2]
        raw_mask = self.tracker.process_frame(frame)
        mask = self._as_uint8_mask(raw_mask)

        # Guardar una nueva captura congelada cada 'capture_interval' frames
        if self.frame_count % max(1, self.capture_interval) == 0 and np.any(mask > 0):
            self.echoes.append((frame.astype(np.float32), mask.copy()))
        self.frame_count += 1

        canvas = frame.astype(np.float32)
        effect_mask = np.zeros((h, w), dtype=np.float32)

        # Dibujar los ecos desde el más antiguo al más reciente
        total = len(self.echoes)
        for idx, (echo_frame, echo_mask) in enumerate(self.echoes):
            age_factor = (idx + 1) / max(1, total)
            alpha = (echo_mask.astype(np.float32) / 255.0) * (self.echo_opacity * age_factor)

            if self.tint_bgr is not None:
                tinted = echo_frame * 0.55 + self.tint_bgr * 0.45
            else:
                tinted = echo_frame

            canvas = canvas * (1.0 - alpha[..., None]) + tinted * alpha[..., None]
            np.maximum(effect_mask, alpha, out=effect_mask)

            if self.line_thickness > 0:
                contours, _ = cv2.findContours(
                    echo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    color = tuple(map(int, self.tint_bgr)) if self.tint_bgr is not None else (255, 255, 255)
                    cv2.drawContours(
                        canvas, contours, -1, color,
                        thickness=self.line_thickness, lineType=cv2.LINE_AA
                    )

        # Mantener al sujeto actual nítido por encima de sus ecos congelados
        curr_alpha = mask.astype(np.float32) / 255.0
        canvas = canvas * (1.0 - curr_alpha[..., None]) + frame.astype(np.float32) * curr_alpha[..., None]
        np.maximum(effect_mask, curr_alpha, out=effect_mask)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(canvas, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(effect_mask * 255.0, 0, 255).astype(np.uint8)
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteStrobeFreezeEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Strobe Freeze", res[:, :, :3])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                effect.reset()
    finally:
        engine.stop()
        cv2.destroyAllWindows()
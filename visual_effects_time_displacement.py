from collections import deque
import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteTimeDisplacementEffect:
    """
    Efecto Desfase Temporal (Time-Displacement):
    - El fondo se muestra en tiempo real.
    - La silueta humana se renderiza con un retraso de 'delay_frames' fotogramas.
    """

    def __init__(self, delay_frames=12, line_thickness=2):
        self.delay_frames = max(1, delay_frames)
        self.line_thickness = line_thickness
        self.history = deque(maxlen=self.delay_frames + 1)
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()
        self.history.clear()

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

        # Guardar el fotograma actual y su máscara en el historial
        self.history.append((frame.copy(), mask.copy()))

        # Recuperar el fotograma más antiguo disponible en la cola
        delayed_frame, delayed_mask = self.history[0]
        alpha_f = (delayed_mask.astype(np.float32) / 255.0)[..., None]

        # Componer la silueta retrasada sobre el fondo en tiempo real
        combined = delayed_frame.astype(np.float32) * alpha_f + frame.astype(np.float32) * (1.0 - alpha_f)
        output_bgr = np.clip(combined, 0, 255).astype(np.uint8)

        if self.line_thickness > 0:
            contours, _ = cv2.findContours(delayed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    output_bgr, contours, -1, (255, 220, 100),
                    thickness=self.line_thickness, lineType=cv2.LINE_AA
                )

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = output_bgr
        out[..., 3] = delayed_mask
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteTimeDisplacementEffect(delay_frames=15)
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                cv2.imshow("Desfase Temporal", res[:, :, :3])
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()
from collections import deque
import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker

class SilhouetteTrailEffect:
    def __init__(self, trail_length=12, decay=0.78):
        """
        Aplica el efecto de estela de color basado en las máscaras extraídas por el tracker.
        """
        self.trail_length = trail_length
        self.decay = decay
        self.masks = deque(maxlen=trail_length)
        
        # Instanciamos el tracker de silueta
        self.tracker = SilhouetteTracker()

    def reset(self):
        """Reinicia el tracker y limpia el historial de estelas."""
        self.tracker.reset()
        self.masks.clear()

    def apply(self, frame):
        """
        Aplica el efecto estela recibiendo el frame completo.
        :return: Imagen procesada RGBA con color y canal alfa.
        """
        # 1. Obtenemos la máscara de la silueta actual desde el tracker
        foreground = self.tracker.process_frame(frame)
        self.masks.append(foreground)

        # 2. Acumulación y decaimiento del rastro (estela)
        trail_alpha = np.zeros(foreground.shape, dtype=np.float32)
        for age, mask in enumerate(reversed(self.masks)):
            weighted_mask = mask.astype(np.float32) * (
                self.decay**age / 255.0
            )
            np.maximum(trail_alpha, weighted_mask, out=trail_alpha)

        # 3. Composición de color de la estela y la silueta en vivo
        current_alpha = foreground.astype(np.float32) / 255.0
        trail_color = np.array((255, 170, 35), dtype=np.float32)  # Color anaranjado/dorado
        
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

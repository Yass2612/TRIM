from collections import deque
import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteTrailEffect:
    def __init__(self, trail_length=12, decay=0.78):
        """
        Efecto sombra a contraluz con estela fantasmal en los bordes,
        manteniendo intacto el fondo real de la cámara.
        """
        self.trail_length = trail_length
        self.decay = decay
        self.masks = deque(maxlen=trail_length)
        self.outlines = deque(maxlen=trail_length)
        self.tracker = SilhouetteTracker()

    def reset(self):
        """Reinicia el tracker y limpia el historial de estelas."""
        self.tracker.reset()
        self.masks.clear()
        self.outlines.clear()

    def apply(self, frame):
        """
        Aplica la sombra oscura en la silueta actual y una estela luminosa
        translúcida en los bordes de los fotogramas anteriores.
        :return: Imagen procesada BGRA uint8 del mismo tamaño que el frame.
        """
        # 1. Máscara suavizada de la silueta actual (0.0 a 1.0)
        foreground = self.tracker.process_frame(frame)
        current_alpha = foreground.astype(np.float32) / 255.0

        # 2. Extraer el borde fino de la silueta actual para la estela etérea
        grad = cv2.morphologyEx(
            foreground, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        grad = cv2.GaussianBlur(grad, (5, 5), 0).astype(np.float32) / 255.0

        self.masks.append(current_alpha)
        self.outlines.append(grad)

        # 3. Acumular rastro suave de sombra y estela de contorno con decaimiento
        ghost_shadow = np.zeros_like(current_alpha)
        ghost_outline = np.zeros_like(current_alpha)

        for age, (m, o) in enumerate(
            zip(reversed(self.masks), reversed(self.outlines))
        ):
            if age == 0:
                continue  # El frame actual va por separado como sombra principal
            weight = self.decay**age
            np.maximum(ghost_shadow, m * weight * 0.35, out=ghost_shadow)
            np.maximum(ghost_outline, o * weight * 0.75, out=ghost_outline)

        # 4. Composición sobre el fondo real:
        #    - Sombra muy oscura en la silueta actual (92 % negro) + rastro tenue
        darkness = np.clip(np.maximum(current_alpha * 0.92, ghost_shadow), 0.0, 1.0)
        result = frame.astype(np.float32) * (1.0 - darkness[..., None])

        #    - Añadir el halo/estela blanquecina translúcida de los movimientos previos
        ghost_tint = np.array([225.0, 230.0, 235.0], dtype=np.float32)
        result = result * (1.0 - ghost_outline[..., None] * 0.55) + (
            ghost_tint * (ghost_outline[..., None] * 0.55)
        )

        output_bgr = np.clip(result, 0, 255).astype(np.uint8)

        # 5. Canal alfa (dónde actúa el efecto) para el deslizador de interfaz_prueba.py
        effect_mask = np.clip(np.maximum(darkness, ghost_outline), 0.0, 1.0)
        alpha = (effect_mask * 255).astype(np.uint8)

        return np.dstack((output_bgr, alpha))


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
import os
import cv2
import numpy as np

# Silenciar logs innecesarios de TensorFlow / MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteTripleEffect:
    """
    Efecto que triplica la silueta humana detectada por SilhouetteTracker.

    Contrato de salida: apply(frame) -> imagen BGRA uint8 del MISMO tamaño que el frame.
      - Silueta Central: Real / Original o Resaltada.
      - Silueta Izquierda: Desplazada horizontalmente con tinte de color (ej. Cian).
      - Silueta Derecha: Desplazada horizontalmente con tinte de color (ej. Magenta).
    """

    def __init__(
        self,
        offset_x=255,
        offset_y=0,
        decay=0.75,
        color_left=(255, 255, 0),    # Cian en BGR
        color_right=(255, 0, 255),   # Magenta en BGR
        color_center=(0, 255, 255),  # Amarillo neón (o None para vídeo real)
        alpha_clones=0.6,
        line_thickness=2,
    ):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.decay = decay
        self.color_left = np.array(color_left, dtype=np.float32)
        self.color_right = np.array(color_right, dtype=np.float32)
        self.color_center = np.array(color_center, dtype=np.float32) if color_center is not None else None
        self.alpha_clones = alpha_clones
        self.line_thickness = line_thickness

        self.trail_left = None
        self.trail_right = None
        self.trail_center = None
        self.tracker = SilhouetteTracker()

    def reset(self):
        """Reinicia el tracker y las matrices de memoria de estela."""
        self.tracker.reset()
        self.trail_left = None
        self.trail_right = None
        self.trail_center = None

    @staticmethod
    def _as_uint8_mask(mask):
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def _shift_mask(self, mask, shift_x, shift_y):
        """Desplaza espacialmente una máscara binaria en los ejes X e Y."""
        h, w = mask.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def apply(self, frame):
        h, w = frame.shape[:2]

        # 1. Obtener la máscara principal en tiempo real
        raw_mask = self.tracker.process_frame(frame)
        mask_center = self._as_uint8_mask(raw_mask)

        # 2. Generar las máscaras desplazadas a la izquierda y derecha
        mask_left = self._shift_mask(mask_center, -self.offset_x, -self.offset_y)
        mask_right = self._shift_mask(mask_center, self.offset_x, self.offset_y)

        # 3. Inicializar o desvanecer estelas temporales
        if self.decay > 0:
            if self.trail_center is None or self.trail_center.shape != mask_center.shape:
                self.trail_center = np.zeros((h, w), dtype=np.float32)
                self.trail_left = np.zeros((h, w), dtype=np.float32)
                self.trail_right = np.zeros((h, w), dtype=np.float32)

            self.trail_center = np.maximum(self.trail_center * self.decay, mask_center.astype(np.float32) / 255.0)
            self.trail_left = np.maximum(self.trail_left * self.decay, mask_left.astype(np.float32) / 255.0)
            self.trail_right = np.maximum(self.trail_right * self.decay, mask_right.astype(np.float32) / 255.0)

            m_center_f = self.trail_center
            m_left_f = self.trail_left
            m_right_f = self.trail_right
        else:
            m_center_f = mask_center.astype(np.float32) / 255.0
            m_left_f = mask_left.astype(np.float32) / 255.0
            m_right_f = mask_right.astype(np.float32) / 255.0

        # 4. Renderizar réplicas
        img_left = m_left_f[..., None] * self.color_left * self.alpha_clones
        img_right = m_right_f[..., None] * self.color_right * self.alpha_clones

        # 5. Renderizar silueta Central
        if self.color_center is not None:
            img_center = m_center_f[..., None] * self.color_center
        else:
            img_center = (frame.astype(np.float32) * m_center_f[..., None])

        # 6. Fusión acumulativa de las tres siluetas
        combined = img_left + img_right + img_center

        # 7. Dibujar bordes definidos (convertimos los colores a tuplas BGR estándar de forma segura)
        if self.line_thickness > 0:
            contours_info = [
                (mask_left, tuple(map(int, self.color_left))),
                (mask_right, tuple(map(int, self.color_right))),
                (mask_center, (255, 255, 255)),
            ]
            for m, color_tuple in contours_info:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(
                        combined, contours, -1, color_tuple,
                        thickness=self.line_thickness, lineType=cv2.LINE_AA
                    )

        bgr = cv2.convertScaleAbs(combined)

        # 8. Canal Alfa
        alpha_mask = np.maximum(np.maximum(m_center_f, m_left_f), m_right_f)
        alpha = cv2.convertScaleAbs(alpha_mask * 255.0)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = bgr
        out[..., 3] = alpha
        return out


def run_effect(engine):
    effect = SilhouetteTripleEffect(offset_x=55, decay=0.70)
    last_processed_frame = None
    processed = None

    cv2.namedWindow("Captura Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Silueta Triplicada", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                frame = frames[-1]
                if frame is not last_processed_frame:
                    processed = effect.apply(frame)
                    last_processed_frame = frame

                cv2.imshow("Captura Original", frame)
                if processed is not None:
                    cv2.imshow("Silueta Triplicada", processed[:, :, :3])

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
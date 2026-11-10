import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteInvertBodyEffect:
    """
    Efecto de inversión espacial:
    - mode="body": Invierte ÚNICAMENTE los colores dentro de tu silueta humana.
    - mode="background": Invierte el FONDO y deja tu cuerpo con sus colores reales.
    """

    def __init__(self, mode="body", line_thickness=2):
        """
        :param mode: "body" (invertir cuerpo) o "background" (invertir fondo).
        :param line_thickness: Grosor de la línea de contorno neón/resaltado.
        """
        self.mode = mode
        self.line_thickness = line_thickness
        self.tracker = SilhouetteTracker()

    def reset(self):
        """Reinicia el tracker."""
        self.tracker.reset()

    @staticmethod
    def _as_uint8_mask(mask):
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def apply(self, frame):
        """
        Devuelve el frame con la silueta invertida y canal Alfa de 4 canales (BGRA).
        """
        h, w = frame.shape[:2]

        # 1. Obtener la máscara binaria en tiempo real (MediaPipe Tasks)
        raw_mask = self.tracker.process_frame(frame)
        mask = self._as_uint8_mask(raw_mask)

        # 2. Invertir totalmente los colores del frame (255 - píxel)
        frame_inverted = cv2.bitwise_not(frame)

        # 3. Recorte diferencial (Cuerpo Invertido vs Fondo Invertido)
        if self.mode == "body":
            # Extraer solo tu cuerpo de la imagen invertida
            body_part = cv2.bitwise_and(frame_inverted, frame_inverted, mask=mask)
            # Extraer el fondo real (lo opuesto a tu cuerpo)
            bg_part = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            # Combinar ambos
            output_bgr = cv2.add(body_part, bg_part)
        else:
            # Fondo Invertido + Cuerpo Real
            bg_part = cv2.bitwise_and(frame_inverted, frame_inverted, mask=cv2.bitwise_not(mask))
            body_part = cv2.bitwise_and(frame, frame, mask=mask)
            output_bgr = cv2.add(body_part, bg_part)

        # 4. Dibujar contorno para delimitar la silueta invertida
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and self.line_thickness > 0:
            cv2.drawContours(output_bgr, contours, -1, (0, 255, 220), thickness=self.line_thickness, lineType=cv2.LINE_AA)

        # 5. Salida de 4 canales (BGRA)
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = output_bgr
        out[..., 3] = 255  # Opacidad completa
        return out


def run_effect(engine):
    # Puedes cambiar mode="body" por mode="background" según lo que busques
    effect = SilhouetteInvertBodyEffect(mode="body", line_thickness=2)
    last_processed_frame = None
    processed = None

    cv2.namedWindow("Captura Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Silueta Invertida", cv2.WINDOW_NORMAL)

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
                    cv2.imshow("Silueta Invertida", processed[:, :, :3])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                # Alternar modo con la tecla 'm'
                effect.mode = "background" if effect.mode == "body" else "body"
                print(f"[INFO] Modo cambiado a: {effect.mode}")
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
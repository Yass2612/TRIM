import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouettePixelEffect:
    """
    Efecto Pixelado / Mosaico Selectivo:
    - Aplica efecto 8-bits únicamente dentro de la silueta ('body') o en el fondo ('background').
    """

    def __init__(self, pixel_size=14, mode="body"):
        self.pixel_size = max(2, pixel_size)
        self.mode = mode
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()

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

        # Reducir y ampliar con interpolación Nearest-Neighbor para crear bloques cuadrados
        small_w = max(1, w // self.pixel_size)
        small_h = max(1, h // self.pixel_size)
        tiny = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)

        active_mask = mask if self.mode == "body" else cv2.bitwise_not(mask)
        alpha_f = (active_mask.astype(np.float32) / 255.0)[..., None]

        combined = pixelated.astype(np.float32) * alpha_f + frame.astype(np.float32) * (1.0 - alpha_f)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(combined, 0, 255).astype(np.uint8)
        out[..., 3] = active_mask
        return out


class SilhouetteEnergyAuraEffect:
    """
    Efecto Aura de Energía / Fuego:
    - Dilata la silueta y aplica distorsión ondulada ascendente con tonos eléctricos/fuego.
    """

    def __init__(self, aura_size=25, color_inner=(0, 255, 255), color_outer=(0, 80, 255)):
        self.aura_size = aura_size
        self.color_inner = np.array(color_inner, dtype=np.float32)
        self.color_outer = np.array(color_outer, dtype=np.float32)
        self.tick = 0
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()
        self.tick = 0

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

        # Dilatar máscara para crear el halo exterior
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.aura_size, self.aura_size))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Desplazar hacia arriba y aplicar onda sinusoidal horizontal
        M = np.float32([[1, 0, 0], [0, 1, -10]])
        shifted = cv2.warpAffine(dilated, M, (w, h), borderValue=0)

        rows = np.arange(h, dtype=np.float32)
        wave = (np.sin(rows * 0.12 + self.tick * 0.35) * 6.0).astype(np.int32)
        wavy_aura = np.zeros_like(shifted)
        for y in range(h):
            wavy_aura[y] = np.roll(shifted[y], wave[y])

        blurred_aura = cv2.GaussianBlur(wavy_aura, (21, 21), 0).astype(np.float32) / 255.0
        body_f = (mask.astype(np.float32) / 255.0)

        # Restar el cuerpo interior para que el aura rodee la silueta
        ring = np.clip(blurred_aura - body_f * 0.85, 0.0, 1.0)[..., None]
        aura_color = self.color_outer * (1.0 - ring) + self.color_inner * ring
        aura_layer = aura_color * ring * 1.4

        self.tick += 1
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(aura_layer, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(ring[..., 0] * 255.0, 0, 255).astype(np.uint8)
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteEnergyAuraEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                frame = frames[-1]
                res = effect.apply(frame)
                a = (res[..., 3:4].astype(np.float32) / 255.0)
                comp = np.clip(frame.astype(np.float32) + res[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)
                cv2.imshow("Aura de Energia", comp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()
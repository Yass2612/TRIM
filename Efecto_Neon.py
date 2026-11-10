import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteNeonEffect:
    """
    Efecto neón sobre la silueta que entrega SilhouetteTracker.

    Contrato de salida (pensado para la interfaz y para combinar con otros
    efectos): apply(frame) -> imagen BGRA uint8 del MISMO tamaño que el frame.
      - BGR: neón sobre fondo negro.
      - A  : intensidad de luz (0 donde no hay nada), útil para componer.
    """

    def __init__(
        self,
        palette_rgb=((0, 255, 220), (110, 90, 255), (255, 40, 200)),
        decay=0.80,
        color_steps=None,
        core_mix=0.75,
        line_thickness=2,
        show_holes=True,
        min_area=400,
        min_hole_area=150,
        glow_layers=((2.0, 1.0), (6.0, 0.8), (14.0, 0.6), (30.0, 0.45)),
        fill_strength=0.08,
        flicker=0.06,
    ):
        """
        :param palette_rgb: colores (RGB) por los que pasa la estela, del más
            reciente al más antiguo. Con un solo color no hay cambio de tono.
        :param decay: desvanecimiento de la estela por frame (0-1).
        :param color_steps: frames que tarda la estela en recorrer toda la
            paleta. None = lo calcula solo, hasta que la intensidad es ~5 %.
        :param core_mix: blanco mezclado en el núcleo de la línea actual (0-1).
        :param line_thickness: grosor del contorno en píxeles.
        :param show_holes: dibuja también los contornos interiores (huecos).
        :param min_area / min_hole_area: áreas mínimas (anti-ruido) para
            contornos exteriores y para huecos.
        :param glow_layers: tuplas (sigma, peso) de los halos gaussianos.
        :param fill_strength: relleno tenue del interior (0 = off).
        :param flicker: parpadeo aleatorio de intensidad (0 = off).
        """
        self.decay = decay
        self.line_thickness = line_thickness
        self.show_holes = show_holes
        self.min_area = min_area
        self.min_hole_area = min_hole_area
        self.fill_strength = fill_strength
        self.flicker = flicker

        if color_steps is None:
            color_steps = int(np.ceil(np.log(0.05) / np.log(decay)))
        self.max_age = int(np.clip(color_steps, 2, 254))

        # Tabla edad -> color BGR (interpolando la paleta). Se hace una sola vez.
        pal = np.array(palette_rgb, dtype=np.float32)[:, ::-1] / 255.0
        src = np.linspace(0.0, 1.0, len(pal))
        dst = np.linspace(0.0, 1.0, self.max_age + 1)
        self._lut = np.stack(
            [np.interp(dst, src, pal[:, c]) for c in range(3)], axis=1
        ).astype(np.float32)
        self.core_bgr = self._lut[0] * (1.0 - core_mix) + core_mix

        # Misma tabla en formato cv2.LUT (256 entradas x 3 canales, float32)
        lut_cv = np.empty((1, 256, 3), dtype=np.float32)
        lut_cv[0, : self.max_age + 1] = self._lut
        lut_cv[0, self.max_age + 1 :] = self._lut[-1]
        self._lut_cv = lut_cv

        # Agrupamos los halos por resolución: los grandes se calculan a
        # resolución reducida (el resultado es igual de suave y mucho más rápido).
        groups = {}
        for sigma, weight in glow_layers:
            scale = 1.0 if sigma < 1.5 else 0.5 if sigma < 12 else 0.25
            groups.setdefault(scale, []).append((sigma * scale, weight))
        self._glow_groups = list(groups.items())

        self.intensity = None  # float32 (H, W): brillo de la estela
        self.age = None        # uint8   (H, W): edad del trazo dominante
        self.tracker = SilhouetteTracker()

    def reset(self):
        """Reinicia el tracker y limpia la estela."""
        self.tracker.reset()
        self.intensity = None
        self.age = None

    # ------------------------------------------------------------------ #
    # Pasos internos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_uint8_mask(mask):
        """Garantiza una máscara 2D uint8 con valores 0-255."""
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def _outline(self, mask):
        """Contorno exterior (y huecos interiores) como línea antialiasada."""
        smooth = cv2.GaussianBlur(mask, (0, 0), 1.5)
        _, binary = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)

        # RETR_CCOMP devuelve contornos exteriores y los huecos que contienen.
        mode = cv2.RETR_CCOMP if self.show_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(
            binary, mode, cv2.CHAIN_APPROX_NONE
        )

        keep = []
        for i, contour in enumerate(contours):
            is_hole = self.show_holes and hierarchy[0][i][3] != -1
            limit = self.min_hole_area if is_hole else self.min_area
            if cv2.contourArea(contour) >= limit:
                keep.append(contour)

        outline = np.zeros_like(binary)
        cv2.drawContours(
            outline, keep, -1, 255,
            thickness=self.line_thickness, lineType=cv2.LINE_AA,
        )
        return outline

    def _update_trail(self, outline):
        """
        Estela recursiva con memoria de edad: cada píxel recuerda su brillo
        y cuántos frames hace que se dibujó. Coste constante por frame.
        """
        if self.intensity is None or self.intensity.shape != outline.shape:
            self.intensity = np.zeros_like(outline)
            self.age = np.full(outline.shape, self.max_age, dtype=np.uint8)

        self.intensity *= self.decay
        self.age += 1
        np.minimum(self.age, self.max_age, out=self.age)

        fresh = outline > self.intensity       # el trazo nuevo manda
        self.intensity[fresh] = outline[fresh]
        self.age[fresh] = 0

    def _soft_fill(self, foreground):
        """Relleno interior tenue, calculado a 1/4 de resolución (es muy suave)."""
        h, w = foreground.shape
        small = cv2.resize(
            foreground, (max(1, w // 4), max(1, h // 4)),
            interpolation=cv2.INTER_AREA,
        )
        small = small.astype(np.float32) * (self.fill_strength / 255.0)
        small = cv2.GaussianBlur(small, (0, 0), 1.0)
        small3 = small[..., None] * self._lut[0]
        return cv2.resize(small3, (w, h), interpolation=cv2.INTER_LINEAR)

    def _glow(self, colored):
        """Halo: suma de desenfoques gaussianos de distinto radio."""
        h, w = colored.shape[:2]
        glow = np.zeros_like(colored)
        for scale, layers in self._glow_groups:
            if scale == 1.0:
                small = colored
            else:
                size = (max(1, int(w * scale)), max(1, int(h * scale)))
                small = cv2.resize(colored, size, interpolation=cv2.INTER_AREA)

            acc = np.zeros_like(small)
            for sigma, weight in layers:
                acc += cv2.GaussianBlur(small, (0, 0), sigma) * weight

            if scale != 1.0:
                acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
            glow += acc
        return glow

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def apply(self, frame):
        """
        :return: BGRA uint8 (neón sobre fondo negro), mismo tamaño que frame.
        """
        foreground = self._as_uint8_mask(self.tracker.process_frame(frame))
        outline = self._outline(foreground).astype(np.float32) * (1.0 / 255.0)
        self._update_trail(outline)

        # Estela coloreada según su edad (cv2.LUT: edad -> color BGR)
        age3 = cv2.merge((self.age, self.age, self.age))
        colored = cv2.LUT(age3, self._lut_cv)
        colored *= self.intensity[..., None]

        # Halo + estela + núcleo caliente (solo en los píxeles del contorno)
        img = self._glow(colored)
        img += colored
        ys, xs = np.nonzero(outline)
        img[ys, xs] += outline[ys, xs, None] * self.core_bgr

        if self.fill_strength > 0:
            img += self._soft_fill(foreground)

        gain = 255.0
        if self.flicker > 0:
            gain *= 1.0 - self.flicker * np.random.rand()

        bgr = cv2.convertScaleAbs(img, alpha=gain)   # escala y satura a uint8
        alpha = np.maximum(np.maximum(bgr[..., 0], bgr[..., 1]), bgr[..., 2])

        out = np.empty((bgr.shape[0], bgr.shape[1], 4), dtype=np.uint8)
        out[..., :3] = bgr
        out[..., 3] = alpha
        return out

    @staticmethod
    def overlay_on_frame(frame, neon_bgra, background_dim=0.35):
        """Utilidad para el futuro: suma el neón sobre el frame oscurecido."""
        base = frame.astype(np.float32) * background_dim
        out = base + neon_bgra[..., :3].astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


def run_effect(engine):
    effect = SilhouetteNeonEffect()
    last_processed_frame = None
    processed = None

    cv2.namedWindow("Captura - Motor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Neon - fondo negro", cv2.WINDOW_NORMAL)

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
                    cv2.imshow("Neon - fondo negro", processed[:, :, :3])

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
"""
combinador_efectos.py

Encadena varios efectos de silueta y los expone como UN SOLO efecto con el
mismo contrato que los demás:

    pipeline.reset()
    pipeline.apply(frame) -> BGRA uint8 del mismo tamaño que el frame

Así la interfaz (interfaz_prueba.py) puede tratarlo como una opción más.

Cómo funciona la cadena
-----------------------
Los efectos se aplican EN ORDEN y cada uno recibe el resultado del anterior:

    cámara -> efecto 1 -> efecto 2 -> ... -> resultado

Cada efecto devuelve BGRA y su resultado se mezcla sobre la imagen anterior
usando su canal alfa (y la opacidad del efecto en la cadena). Es la misma
mezcla que ya hace la interfaz, por eso funciona igual con los efectos que
devuelven el frame entero (Estela, Inversión) que con los que devuelven una
capa de luz sobre negro (Neón, Triplicada).

El orden importa: [inversion, neon] dibuja el neón sobre el cuerpo invertido;
[neon, inversion] invierte también los colores del neón dentro del cuerpo.

Una sola segmentación por frame
-------------------------------
Cada efecto crea su propio SilhouetteTracker, así que 4 efectos cargarían
4 modelos y harían 4 inferencias por frame. El combinador los sustituye por
un único SharedSilhouetteTracker que segmenta el frame ORIGINAL de la cámara
una vez y reparte la misma máscara a todos. Gracias a eso, los efectos pueden
procesar la imagen ya modificada por el anterior sin que la segmentación se
confunda.

Uso desde código
----------------
    pipeline = EffectPipeline(["inversion", "neon"])
    pipeline.toggle("triple")                 # añade/quita al final
    pipeline.set_opacity("neon", 0.7)
    pipeline.move("neon", 0)                  # cambia el orden
    pipeline.get("triple").offset_x = 200     # ajustar parámetros del efecto

Uso en solitario (ventanas OpenCV)
----------------------------------
    python combinador_efectos.py                      # cadena vacía
    python combinador_efectos.py inversion neon       # cadena inicial
"""
from __future__ import annotations

import argparse
import importlib
import time
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------- #
# Registro de efectos disponibles
# ---------------------------------------------------------------------- #
@dataclass
class EffectSpec:
    label: str            # nombre para mostrar en la interfaz
    module: str           # módulo donde vive el efecto
    class_name: str       # clase del efecto
    kwargs: dict = field(default_factory=dict)  # parámetros por defecto


# Para añadir un efecto nuevo basta con registrarlo aquí. Solo necesita:
#   - __init__ que cree self.tracker = SilhouetteTracker()
#   - reset()
#   - apply(frame) -> BGRA uint8
EFFECT_REGISTRY = {
    "neon": EffectSpec("Neón", "Efecto_Neon", "SilhouetteNeonEffect"),
    "estela": EffectSpec("Estela", "visual_effects", "SilhouetteTrailEffect"),
    "triple": EffectSpec(
        "Triplicada", "visual_effects_triple", "SilhouetteTripleEffect",
        {"offset_x": 150, "decay": 0.70},
    ),
    "inversion": EffectSpec(
        "Inversión", "visual_effects_inversion_silueta",
        "SilhouetteInvertBodyEffect",
        {"mode": "body", "line_thickness": 2},
    ),
}


# ---------------------------------------------------------------------- #
# Tracker compartido
# ---------------------------------------------------------------------- #
class SharedSilhouetteTracker:
    """
    Hace de SilhouetteTracker para todos los efectos de la cadena.

    El combinador llama a begin_frame() con el frame original de la cámara;
    a partir de ahí, process_frame() devuelve esa misma máscara (una copia)
    sin importar qué frame reciba, hasta que se llame a end_frame().
    """

    def __init__(self, tracker=None):
        if tracker is None:
            # Import tardío: el módulo se puede importar sin MediaPipe, y los
            # errores (modelo ausente, MediaPipe sin instalar) saltan aquí, al
            # crear el combinador, igual que al crear cualquier otro efecto.
            from silhouette_tracker import SilhouetteTracker

            tracker = SilhouetteTracker()
        self._tracker = tracker
        self._mask = None

    def begin_frame(self, camera_frame):
        self._mask = self._tracker.process_frame(camera_frame)

    def end_frame(self):
        self._mask = None

    def process_frame(self, frame):
        if self._mask is None:  # uso fuera de la cadena: comportamiento normal
            return self._tracker.process_frame(frame)
        return self._mask.copy()

    def reset(self):
        self._mask = None
        self._tracker.reset()


# ---------------------------------------------------------------------- #
# Combinador
# ---------------------------------------------------------------------- #
@dataclass
class Slot:
    key: str
    effect: object
    opacity: float = 1.0


class EffectPipeline:
    def __init__(self, chain=(), tracker=None):
        """
        :param chain: claves de EFFECT_REGISTRY, en orden de aplicación.
        :param tracker: tracker ya creado (opcional); si no, se crea uno.
        """
        self.shared = SharedSilhouetteTracker(tracker)
        self.slots: list[Slot] = []
        self.timings: dict[str, float] = {}   # ms por fase del último frame
        self._instances: dict[str, object] = {}
        for key in chain:
            self.add(key)

    # ------------------------- gestión de la cadena ------------------- #
    @property
    def chain(self):
        """Claves de los efectos activos, en orden de aplicación."""
        return [slot.key for slot in self.slots]

    def describe(self):
        """Texto tipo 'Inversión → Neón' (con la opacidad si no es 100 %)."""
        if not self.slots:
            return "(vacía)"
        parts = []
        for slot in self.slots:
            label = EFFECT_REGISTRY[slot.key].label
            if slot.opacity < 1.0:
                label += f" {slot.opacity:.0%}"
            parts.append(label)
        return " → ".join(parts)

    def _slot(self, key):
        for slot in self.slots:
            if slot.key == key:
                return slot
        raise KeyError(f"'{key}' no está en la cadena: {self.chain}")

    def _instance(self, key, overrides=None):
        """Crea (una sola vez) el efecto y lo conecta al tracker compartido."""
        if key not in EFFECT_REGISTRY:
            raise KeyError(
                f"Efecto desconocido '{key}'. Disponibles: {list(EFFECT_REGISTRY)}"
            )
        overrides = overrides or {}

        if key in self._instances:
            effect = self._instances[key]
            for name, value in overrides.items():
                if not hasattr(effect, name):
                    raise AttributeError(f"'{key}' no tiene el parámetro '{name}'")
                setattr(effect, name, value)
            return effect

        spec = EFFECT_REGISTRY[key]
        module = importlib.import_module(spec.module)
        effect_class = getattr(module, spec.class_name)

        # Mientras se construye, el módulo "ve" el tracker compartido en lugar
        # de la clase real: así no se carga un modelo de MediaPipe por efecto.
        original = getattr(module, "SilhouetteTracker", None)
        module.SilhouetteTracker = lambda *args, **kwargs: self.shared
        try:
            effect = effect_class(**{**spec.kwargs, **overrides})
        finally:
            if original is not None:
                module.SilhouetteTracker = original
        effect.tracker = self.shared  # por si el efecto lo crea de otra forma

        self._instances[key] = effect
        return effect

    def add(self, key, opacity=1.0, index=None, **overrides):
        """Añade un efecto (al final, o en `index`). Devuelve su Slot."""
        if key in self.chain:
            raise ValueError(f"'{key}' ya está en la cadena")
        effect = self._instance(key, overrides)
        effect.reset()  # empieza sin estela de usos anteriores
        slot = Slot(key, effect, float(np.clip(opacity, 0.0, 1.0)))
        if index is None:
            self.slots.append(slot)
        else:
            self.slots.insert(index, slot)
        return slot

    def remove(self, key):
        self.slots.remove(self._slot(key))

    def toggle(self, key, **overrides):
        """Quita el efecto si está en la cadena, o lo añade al final si no.
        Devuelve True si queda activo."""
        if key in self.chain:
            self.remove(key)
            return False
        self.add(key, **overrides)
        return True

    def move(self, key, new_index):
        slot = self._slot(key)
        self.slots.remove(slot)
        self.slots.insert(new_index, slot)

    def set_opacity(self, key, opacity):
        self._slot(key).opacity = float(np.clip(opacity, 0.0, 1.0))

    def clear(self):
        self.slots.clear()

    def get(self, key):
        """El objeto del efecto, para ajustar sus parámetros en vivo
        (por ejemplo pipeline.get('triple').offset_x = 200)."""
        return self._instance(key)

    # ------------------------- contrato de efecto --------------------- #
    def reset(self):
        for effect in self._instances.values():
            effect.reset()
        self.shared.reset()

    @staticmethod
    def _blend(base, overlay, opacity):
        """Mezcla `overlay` (BGRA) sobre `base` (BGR) con su alfa y opacidad."""
        if overlay.ndim == 2 or overlay.shape[2] == 3:
            alpha_u8 = None
            overlay_bgr = overlay if overlay.ndim == 3 else cv2.cvtColor(
                overlay, cv2.COLOR_GRAY2BGR
            )
        else:
            alpha_u8 = overlay[..., 3]
            overlay_bgr = overlay[..., :3]

        # Capa totalmente opaca (p. ej. Inversión) al 100 %: sustituye sin mezclar
        if opacity >= 1.0 and (alpha_u8 is None or alpha_u8.min() == 255):
            return np.ascontiguousarray(overlay_bgr)

        if alpha_u8 is None:
            alpha = np.float32(opacity)
        else:
            alpha = (alpha_u8.astype(np.float32) * (opacity / 255.0))[..., None]
        base_f = base.astype(np.float32)
        out = base_f + (overlay_bgr.astype(np.float32) - base_f) * alpha
        return out.astype(np.uint8)

    def apply(self, frame):
        """
        :param frame: frame BGR uint8 de la cámara.
        :return: BGRA uint8 con el resultado de toda la cadena (alfa = 255).
                 En la interfaz, el deslizador de intensidad hace de fundido
                 entre el vídeo original y el resultado combinado.
        """
        if not self.slots:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        t0 = time.perf_counter()
        self.shared.begin_frame(frame)          # una sola segmentación
        self.timings["mascara"] = (time.perf_counter() - t0) * 1000.0

        running = frame
        try:
            for slot in self.slots:
                t0 = time.perf_counter()
                overlay = slot.effect.apply(running)
                running = self._blend(running, overlay, slot.opacity)
                self.timings[slot.key] = (time.perf_counter() - t0) * 1000.0
        finally:
            self.shared.end_frame()

        return cv2.cvtColor(running, cv2.COLOR_BGR2BGRA)


# ---------------------------------------------------------------------- #
# Modo en solitario: ventanas OpenCV para probar combinaciones en directo
# ---------------------------------------------------------------------- #
def _ascii(text):
    """cv2.putText solo pinta ASCII: quita tildes y cambia las flechas."""
    text = text.replace("→", ">")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _draw_hud(image, pipeline, ms):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
    fps = 1000.0 / ms if ms > 0 else 0.0
    cv2.putText(
        image, _ascii(f"Cadena: {pipeline.describe()}   [{ms:.0f} ms | {fps:.0f} FPS]"),
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 220), 1, cv2.LINE_AA,
    )
    keys = "  ".join(
        f"{i}:{_ascii(spec.label)}"
        for i, spec in enumerate(EFFECT_REGISTRY.values(), start=1)
    )
    cv2.putText(
        image, f"{keys} | +/- opacidad | c vaciar | r reset | q salir",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return image


def run_combiner(engine, initial_chain=()):
    pipeline = EffectPipeline(initial_chain)
    effect_keys = list(EFFECT_REGISTRY)

    last_id = None
    processed = None
    ms = 0.0
    redraw = True

    cv2.namedWindow("Captura Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Combinador", cv2.WINDOW_NORMAL)
    print("[INFO] Pulsa 1-%d para activar/desactivar efectos: se aplican en el "
          "orden en que los actives." % len(effect_keys))

    try:
        while True:
            frame_id, frame = engine.get_latest_frame_with_id()
            if frame is not None and frame_id != last_id:
                t0 = time.perf_counter()
                processed = pipeline.apply(frame)[..., :3]
                ms = (time.perf_counter() - t0) * 1000.0
                last_id = frame_id
                cv2.imshow("Captura Original", frame)
                redraw = True

            if processed is not None and redraw:
                cv2.imshow("Combinador", _draw_hud(processed.copy(), pipeline, ms))
                redraw = False

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            redraw = True

            if key == ord("q"):
                break
            if key == ord("r"):
                pipeline.reset()
            elif key == ord("c"):
                pipeline.clear()
            elif ord("1") <= key < ord("1") + len(effect_keys):
                name = effect_keys[key - ord("1")]
                try:
                    state = "activado" if pipeline.toggle(name) else "desactivado"
                    print(f"[INFO] {EFFECT_REGISTRY[name].label} {state} -> "
                          f"{pipeline.describe()}")
                except Exception as error:  # módulo ausente, modelo, etc.
                    print(f"[ERROR] No se pudo cargar '{name}': {error}")
            elif key in (ord("+"), ord("="), ord("-")) and pipeline.slots:
                last = pipeline.slots[-1]
                step = -0.1 if key == ord("-") else 0.1
                pipeline.set_opacity(last.key, last.opacity + step)
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Combina efectos de silueta en directo."
    )
    parser.add_argument(
        "effects", nargs="*",
        help=f"Cadena inicial, en orden. Disponibles: {', '.join(EFFECT_REGISTRY)}",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    unknown = [name for name in args.effects if name not in EFFECT_REGISTRY]
    if unknown:
        parser.error(f"efectos desconocidos: {unknown}")

    from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded

    engine = VideoEngineThreaded(
        src=args.camera, width=640, height=480, buffer_size=30
    ).start()
    try:
        run_combiner(engine, args.effects)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
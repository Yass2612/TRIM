"""
combinador_efectos.py

Encadena varios efectos de silueta y los expone como UN SOLO efecto.
"""
from __future__ import annotations

import argparse
import importlib
import time
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class EffectSpec:
    label: str
    module: str
    class_name: str
    kwargs: dict = field(default_factory=dict)


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
    "strobe": EffectSpec(
        "Strobe", "visual_effects_strobe",
        "SilhouetteStrobeFreezeEffect",
        {"capture_interval": 24, "max_echoes": 5},
    ),
    "calor": EffectSpec("Calor", "visual_effects_heatmap", "SilhouetteHeatmapEffect"),
    "comic": EffectSpec("Cómic", "visual_effects_comic", "SilhouetteComicEffect"),
    "desfase": EffectSpec(
        "Desfase", "visual_effects_time_displacement",
        "SilhouetteTimeDisplacementEffect",
        {"delay_frames": 12},
    ),
    "particulas": EffectSpec("Chispas", "visual_effects_particles", "SilhouetteParticlesEffect"),
    "bokeh": EffectSpec(
        "Retrato", "visual_effects_bokeh",
        "SilhouetteBokehEffect",
        {"blur_strength": 35},
    ),
    "fondo": EffectSpec("Fondo Virtual", "visual_effects_virtual_bg", "SilhouetteVirtualBgEffect"),
    "pixel": EffectSpec(
        "Pixelado", "visual_effects_pixel_aura",
        "SilhouettePixelEffect",
        {"pixel_size": 14, "mode": "body"},
    ),
    "aura": EffectSpec("Aura", "visual_effects_pixel_aura", "SilhouetteEnergyAuraEffect"),
}


class SharedSilhouetteTracker:
    def __init__(self, tracker=None):
        if tracker is None:
            from silhouette_tracker import SilhouetteTracker
            tracker = SilhouetteTracker()
        self._tracker = tracker
        self._mask = None

    def begin_frame(self, camera_frame):
        self._mask = self._tracker.process_frame(camera_frame)

    def end_frame(self):
        self._mask = None

    def process_frame(self, frame):
        if self._mask is None:
            return self._tracker.process_frame(frame)
        return self._mask.copy()

    def reset(self):
        self._mask = None
        self._tracker.reset()


@dataclass
class Slot:
    key: str
    effect: object
    opacity: float = 1.0


class EffectPipeline:
    def __init__(self, chain=(), tracker=None):
        self.shared = SharedSilhouetteTracker(tracker)
        self.slots: list[Slot] = []
        self.timings: dict[str, float] = {}
        self._instances: dict[str, object] = {}
        for key in chain:
            self.add(key)

    @property
    def chain(self):
        return [slot.key for slot in self.slots]

    def describe(self):
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
        raise KeyError(f"'{key}' no está en la cadena")

    def _instance(self, key, overrides=None):
        if key not in EFFECT_REGISTRY:
            raise KeyError(f"Efecto desconocido '{key}'.")
        overrides = overrides or {}

        if key in self._instances:
            effect = self._instances[key]
            for name, value in overrides.items():
                if hasattr(effect, name):
                    setattr(effect, name, value)
            return effect

        spec = EFFECT_REGISTRY[key]
        module = importlib.import_module(spec.module)
        effect_class = getattr(module, spec.class_name)

        original = getattr(module, "SilhouetteTracker", None)
        module.SilhouetteTracker = lambda *args, **kwargs: self.shared
        try:
            effect = effect_class(**{**spec.kwargs, **overrides})
        finally:
            if original is not None:
                module.SilhouetteTracker = original
        effect.tracker = self.shared

        self._instances[key] = effect
        return effect

    def add(self, key, opacity=1.0, index=None, **overrides):
        if key in self.chain:
            raise ValueError(f"'{key}' ya está en la cadena")
        effect = self._instance(key, overrides)
        effect.reset()
        slot = Slot(key, effect, float(np.clip(opacity, 0.0, 1.0)))
        if index is None:
            self.slots.append(slot)
        else:
            self.slots.insert(index, slot)
        return slot

    def remove(self, key):
        self.slots.remove(self._slot(key))

    def toggle(self, key, **overrides):
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
        return self._instance(key)

    def reset(self):
        for effect in self._instances.values():
            effect.reset()
        self.shared.reset()

    @staticmethod
    def _blend(base, overlay, opacity):
        if overlay.ndim == 2 or overlay.shape[2] == 3:
            alpha_u8 = None
            overlay_bgr = overlay if overlay.ndim == 3 else cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        else:
            alpha_u8 = overlay[..., 3]
            overlay_bgr = overlay[..., :3]

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
        if not self.slots:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        t0 = time.perf_counter()
        self.shared.begin_frame(frame)
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


SHORTCUT_KEYS = "123456789abcd"

def _ascii(text):
    text = text.replace("→", ">")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def _draw_hud(image, pipeline, ms):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 84), (0, 0, 0), -1)
    fps = 1000.0 / ms if ms > 0 else 0.0
    cv2.putText(
        image, _ascii(f"Cadena: {pipeline.describe()}   [{ms:.0f} ms | {fps:.0f} FPS]"),
        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 220), 1, cv2.LINE_AA,
    )

    items = [
        f"{SHORTCUT_KEYS[i]}:{_ascii(spec.label)}"
        for i, spec in enumerate(EFFECT_REGISTRY.values())
        if i < len(SHORTCUT_KEYS)
    ]
    row1 = "  ".join(items[:5])
    row2 = "  ".join(items[5:10])
    row3 = "  ".join(items[10:]) + " | +/- opac | x vaciar | r reset | q salir"

    cv2.putText(image, row1, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, row2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, row3, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return image

def run_combiner(engine, initial_chain=()):
    pipeline = EffectPipeline(initial_chain)
    effect_keys = list(EFFECT_REGISTRY)
    key_map = {ord(SHORTCUT_KEYS[i]): k for i, k in enumerate(effect_keys) if i < len(SHORTCUT_KEYS)}

    last_id = None
    processed = None
    ms = 0.0
    redraw = True

    cv2.namedWindow("Captura Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Combinador", cv2.WINDOW_NORMAL)
    print("[INFO] Pulsa 1-9 / a-d para activar o desactivar efectos en cadena.")

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
            elif key == ord("x"):
                pipeline.clear()
            elif key in key_map:
                name = key_map[key]
                try:
                    pipeline.toggle(name)
                except Exception as error:
                    print(f"[ERROR] No se pudo cargar '{name}': {error}")
            elif key in (ord("+"), ord("="), ord("-")) and pipeline.slots:
                last = pipeline.slots[-1]
                step = -0.1 if key == ord("-") else 0.1
                pipeline.set_opacity(last.key, last.opacity + step)
    finally:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Combina efectos de silueta en directo.")
    parser.add_argument("effects", nargs="*", help=f"Cadena inicial, disponibles: {', '.join(EFFECT_REGISTRY)}")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    unknown = [name for name in args.effects if name not in EFFECT_REGISTRY]
    if unknown:
        parser.error(f"efectos desconocidos: {unknown}")

    # Importación corregida apuntando a la carpeta Motor_Y_Captura
    from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
    
    engine = VideoEngineThreaded(src=args.camera, width=640, height=480, buffer_size=30).start()
    try:
        run_combiner(engine, args.effects)
    finally:
        engine.stop()

if __name__ == "__main__":
    main()
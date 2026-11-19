from __future__ import annotations

import base64
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import cv2

# Importación corregida apuntando a la carpeta Motor_Y_Captura
try:
    from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
except ImportError:
    # Por si están en otra carpeta
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded

if TYPE_CHECKING:
    from combinador_efectos import EffectPipeline


class VideoInterface:
    BACKGROUND = "#10131c"
    PANEL = "#191e2a"
    MUTED = "#929bad"
    ACCENT = "#7c6cff"

    EFFECT_ITEMS = [
        ("Original", None),
        ("Estela", "estela"),
        ("Neón", "neon"),
        ("Triplicada", "triple"),
        ("Inversión", "inversion"),
        ("Strobe", "strobe"),
        ("Calor", "calor"),
        ("Cómic", "comic"),
        ("Desfase Temporal", "desfase"),
        ("Chispas / Partículas", "particulas"),
        ("Retrato Bokeh", "bokeh"),
        ("Fondo Virtual", "fondo"),
        ("Pixelado", "pixel"),
        ("Aura Energía", "aura"),
        ("Combinador (Cadena)", "pipeline"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("VIDEO LAB | Procesamiento de vídeo en tiempo real")
        self.root.geometry("1220x860")
        self.root.minsize(1040, 820)
        self.root.configure(bg=self.BACKGROUND)

        self.engine = None
        self.preview_image = None
        self.current_frame = None
        self.video_writer = None
        self._last_frame_id = None

        self.label_to_key = dict(self.EFFECT_ITEMS)
        self.key_to_label = {k: v for v, k in self.EFFECT_ITEMS if k is not None}

        self.selected_mode = tk.StringVar(value="Original")
        self.selected_camera = tk.StringVar(value="0")
        self.mirror_enabled = tk.BooleanVar(value=False)

        self.target_zone = tk.StringVar(value="body")
        self.triple_offset = tk.IntVar(value=150)
        self.strobe_interval = tk.IntVar(value=24)

        self.camera_status = tk.StringVar(value="Conectando con la cámara...")
        self.fps_text = tk.StringVar(value="-- FPS · -- × --")
        self.intensity_text = tk.StringVar(value="100 %")
        self.offset_text = tk.StringVar(value="150 px")
        self.strobe_text = tk.StringVar(value="24 frames")
        self.chain_display = tk.StringVar(value="Cadena: (vacía)")
        self.recording_status = tk.StringVar(value="Grabación detenida")
        self.performance_text = tk.StringVar(value="Buffer: --/30")

        self.effects_cache: dict[str, object] = {}
        self.pipeline: EffectPipeline | None = None
        self.chain_vars: dict[str, tk.BooleanVar] = {}

        self._fps_frames = 0
        self._fps_started = 0.0
        self._display_fps = 0.0

        self._configure_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._start_camera()
        self.root.after(20, self._update_preview)

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Test.Horizontal.TScale",
            background=self.PANEL,
            troughcolor="#303747",
            sliderthickness=16,
        )

    def _build_layout(self):
        header = tk.Frame(self.root, bg=self.BACKGROUND)
        header.pack(fill="x", padx=28, pady=(18, 12))

        tk.Label(header, text="VIDEO LAB", bg=self.BACKGROUND, fg="#a99fff", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(header, text="Procesamiento de vídeo en tiempo real", bg=self.BACKGROUND, fg="#f4f5fa", font=("Segoe UI", 22, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(header, text="Catálogo completo de efectos y combinador en cadena", bg=self.BACKGROUND, fg=self.MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))

        content = tk.Frame(self.root, bg=self.BACKGROUND)
        content.pack(fill="both", expand=True, padx=28, pady=(0, 20))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Panel Izquierdo
        preview_panel = tk.Frame(content, bg=self.PANEL, padx=16, pady=16)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        preview_panel.rowconfigure(1, weight=1)
        preview_panel.columnconfigure(0, weight=1)

        preview_header = tk.Frame(preview_panel, bg=self.PANEL)
        preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(preview_header, text="VISTA DE CÁMARA", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(preview_header, textvariable=self.fps_text, bg="#252b39", fg="#bcb4ff", padx=10, pady=5, font=("Segoe UI", 9, "bold")).pack(side="right")

        self.preview = tk.Label(preview_panel, text="Esperando señal de vídeo...", bg="#0b0e14", fg=self.MUTED, font=("Segoe UI", 12))
        self.preview.grid(row=1, column=0, sticky="nsew")

        status_row = tk.Frame(preview_panel, bg=self.PANEL)
        status_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.status_dot = tk.Label(status_row, text="●", bg=self.PANEL, fg="#e8b451", font=("Segoe UI", 10))
        self.status_dot.pack(side="left", padx=(0, 7))
        tk.Label(status_row, textvariable=self.camera_status, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9), anchor="w").pack(side="left", fill="x", expand=True)

        # Panel Derecho
        controls = tk.Frame(content, bg=self.PANEL, width=320, padx=18, pady=16)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.grid_propagate(False)

        tk.Label(controls, text="CONTROLES", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        tk.Label(controls, text="CÁMARA", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 4))
        self.camera_selector = ttk.Combobox(controls, textvariable=self.selected_camera, values=tuple(str(i) for i in range(6)), state="readonly")
        self.camera_selector.pack(fill="x", pady=(0, 10))
        self.camera_selector.bind("<<ComboboxSelected>>", self._change_camera)

        tk.Label(controls, text="MODO / EFECTO VISUAL", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 4))
        self.effect_selector = ttk.Combobox(controls, textvariable=self.selected_mode, values=[lbl for lbl, _ in self.EFFECT_ITEMS], state="readonly", font=("Segoe UI", 10))
        self.effect_selector.pack(fill="x", pady=(0, 10))
        self.effect_selector.bind("<<ComboboxSelected>>", self._on_mode_changed)

        tk.Label(controls, text="INTENSIDAD / MEZCLA", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 4))
        self.intensity = tk.DoubleVar(value=100)
        self.intensity_scale = ttk.Scale(controls, from_=0, to=100, variable=self.intensity, style="Test.Horizontal.TScale", command=self._update_intensity_label)
        self.intensity_scale.pack(fill="x")
        self.intensity_scale.configure(state="disabled")
        tk.Label(controls, textvariable=self.intensity_text, bg=self.PANEL, fg="#bcb4ff", font=("Segoe UI", 8, "bold")).pack(anchor="e", pady=(2, 4))

        # Contenedores Dinámicos
        self.dynamic_frame = tk.Frame(controls, bg=self.PANEL)
        self.dynamic_frame.pack(fill="x", pady=(4, 4))

        self.triple_frame = tk.Frame(self.dynamic_frame, bg=self.PANEL)
        tk.Label(self.triple_frame, text="SEPARACIÓN CLONES", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 4))
        ttk.Scale(self.triple_frame, from_=0, to=300, variable=self.triple_offset, style="Test.Horizontal.TScale", command=self._update_triple_offset).pack(fill="x")
        tk.Label(self.triple_frame, textvariable=self.offset_text, bg=self.PANEL, fg="#bcb4ff", font=("Segoe UI", 8, "bold")).pack(anchor="e")

        self.zone_frame = tk.Frame(self.dynamic_frame, bg=self.PANEL)
        tk.Label(self.zone_frame, text="ZONA DE APLICACIÓN", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 4))
        zone_btns = tk.Frame(self.zone_frame, bg=self.PANEL)
        zone_btns.pack(fill="x")
        for lbl, val in (("Cuerpo", "body"), ("Fondo", "background")):
            tk.Radiobutton(
                zone_btns, text=lbl, value=val, variable=self.target_zone, command=self._update_target_zone,
                indicatoron=False, padx=8, pady=5, bg="#222837", fg="#e5e8f0", selectcolor=self.ACCENT,
                activebackground="#30374a", activeforeground="white", font=("Segoe UI", 9), relief="flat", cursor="hand2"
            ).pack(side="left", expand=True, fill="x", padx=2)

        self.strobe_frame = tk.Frame(self.dynamic_frame, bg=self.PANEL)
        tk.Label(self.strobe_frame, text="INTERVALO DE ECOS", bg=self.PANEL, fg="#f4f5fa", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 4))
        ttk.Scale(self.strobe_frame, from_=5, to=45, variable=self.strobe_interval, style="Test.Horizontal.TScale", command=self._update_strobe_interval).pack(fill="x")
        tk.Label(self.strobe_frame, textvariable=self.strobe_text, bg=self.PANEL, fg="#bcb4ff", font=("Segoe UI", 8, "bold")).pack(anchor="e")

        self.pipeline_frame = tk.Frame(self.dynamic_frame, bg="#222837", padx=10, pady=8)
        tk.Label(self.pipeline_frame, text="ACTIVA EFECTOS EN CADENA", bg="#222837", fg="#f4f5fa", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))
        grid_frame = tk.Frame(self.pipeline_frame, bg="#222837")
        grid_frame.pack(fill="x")
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        combinable_items = [(lbl, key) for lbl, key in self.EFFECT_ITEMS if key not in (None, "pipeline")]
        for idx, (lbl, key) in enumerate(combinable_items):
            var = tk.BooleanVar(value=False)
            self.chain_vars[key] = var
            cb = tk.Checkbutton(
                grid_frame, text=lbl, variable=var, command=lambda k=key: self._toggle_pipeline_effect(k),
                bg="#222837", fg="#e5e8f0", activebackground="#222837", activeforeground="white",
                selectcolor="#303747", font=("Segoe UI", 8), anchor="w"
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky="w", pady=1)

        tk.Label(self.pipeline_frame, textvariable=self.chain_display, bg="#222837", fg="#a99fff", font=("Segoe UI", 8, "bold"), wraplength=260, justify="left").pack(anchor="w", pady=(6, 4))
        tk.Button(self.pipeline_frame, text="Vaciar cadena", command=self._clear_pipeline, bg="#303747", fg="white", relief="flat", font=("Segoe UI", 8), pady=3, cursor="hand2").pack(fill="x")

        # Botones Inferiores
        tk.Checkbutton(controls, text="Espejo horizontal", variable=self.mirror_enabled, bg=self.PANEL, fg="#e5e8f0", activebackground=self.PANEL, activeforeground="white", selectcolor="#303747", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.effect_note = tk.Label(controls, text="Modo original · señal sin filtros", bg=self.PANEL, fg="#bcb4ff", font=("Segoe UI", 8), wraplength=275, justify="left")
        self.effect_note.pack(anchor="w", pady=(6, 4))
        tk.Label(controls, textvariable=self.performance_text, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8), anchor="w").pack(fill="x", pady=(0, 2))
        tk.Label(controls, textvariable=self.recording_status, bg=self.PANEL, fg="#ee6b72", font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", pady=(0, 4))

        self.capture_button = tk.Button(controls, text="Guardar foto", command=self._save_snapshot, bg="#303747", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), padx=10, pady=6, cursor="hand2")
        self.capture_button.pack(fill="x", side="bottom", pady=(4, 0))
        self.record_button = tk.Button(controls, text="Iniciar grabación", command=self._toggle_recording, bg="#bd4651", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), padx=10, pady=6, cursor="hand2")
        self.record_button.pack(fill="x", side="bottom", pady=(0, 4))
        self.camera_button = tk.Button(controls, text="Detener cámara", command=self._toggle_camera, bg=self.ACCENT, fg="white", relief="flat", font=("Segoe UI", 10, "bold"), padx=12, pady=8, cursor="hand2")
        self.camera_button.pack(fill="x", side="bottom")

    def _get_pipeline(self) -> EffectPipeline:
        if self.pipeline is None:
            from combinador_efectos import EffectPipeline
            self.pipeline = EffectPipeline()
        return self.pipeline

    def _get_single_effect(self, key: str):
        if key not in self.effects_cache:
            import importlib
            from combinador_efectos import EFFECT_REGISTRY
            spec = EFFECT_REGISTRY[key]
            module = importlib.import_module(spec.module)
            cls = getattr(module, spec.class_name)
            self.effects_cache[key] = cls(**spec.kwargs)
        return self.effects_cache[key]

    def _reset_all_effects(self):
        for eff in self.effects_cache.values():
            eff.reset()
        if self.pipeline is not None:
            self.pipeline.reset()

    def _hide_dynamic_panels(self):
        self.triple_frame.pack_forget()
        self.zone_frame.pack_forget()
        self.strobe_frame.pack_forget()
        self.pipeline_frame.pack_forget()

    def _on_mode_changed(self, _event=None):
        label = self.selected_mode.get()
        key = self.label_to_key.get(label)

        try:
            if key == "pipeline":
                self._get_pipeline()
            elif key is not None:
                self._get_single_effect(key)
        except Exception as error:
            self.selected_mode.set("Original")
            self._hide_dynamic_panels()
            self.intensity_scale.configure(state="disabled")
            messagebox.showerror("Error al cargar efecto", str(error))
            return

        self._reset_all_effects()
        self._hide_dynamic_panels()
        self.intensity_scale.configure(state="disabled" if key is None else "normal")

        if key is None:
            self.effect_note.configure(text="Modo original · señal sin filtros")
        elif key == "pipeline":
            self.pipeline_frame.pack(fill="x")
            self.effect_note.configure(text="Encadena múltiples filtros en el orden activado")
        elif key == "triple":
            self.triple_frame.pack(fill="x")
            self._update_triple_offset(self.triple_offset.get())
            self.effect_note.configure(text="Silueta triplicada · ajusta la separación lateral")
        elif key in ("inversion", "pixel"):
            self.zone_frame.pack(fill="x")
            self._update_target_zone()
            self.effect_note.configure(text=f"{label} · elige si afecta al cuerpo o al fondo")
        elif key == "strobe":
            self.strobe_frame.pack(fill="x")
            self._update_strobe_interval(self.strobe_interval.get())
            self.effect_note.configure(text="Strobe Freeze · ecos congelados en el espacio")
        else:
            self.effect_note.configure(text=f"Filtro {label} activo · intensidad ajustable")

    def _toggle_pipeline_effect(self, key: str):
        try:
            pipe = self._get_pipeline()
            pipe.toggle(key)
            self.chain_display.set(f"Cadena: {pipe.describe()}")
        except Exception as error:
            self.chain_vars[key].set(False)
            messagebox.showerror("Error en combinador", f"No se pudo activar '{key}': {error}")

    def _clear_pipeline(self):
        if self.pipeline is not None:
            self.pipeline.clear()
        for var in self.chain_vars.values():
            var.set(False)
        self.chain_display.set("Cadena: (vacía)")

    def _update_intensity_label(self, value):
        self.intensity_text.set(f"{float(value):.0f} %")

    def _update_triple_offset(self, value):
        offset = int(float(value))
        self.offset_text.set(f"{offset} px")
        if "triple" in self.effects_cache:
            self.effects_cache["triple"].offset_x = offset

    def _update_strobe_interval(self, value):
        interval = int(float(value))
        self.strobe_text.set(f"{interval} frames")
        if "strobe" in self.effects_cache:
            self.effects_cache["strobe"].capture_interval = interval

    def _update_target_zone(self):
        zone = self.target_zone.get()
        if "inversion" in self.effects_cache:
            self.effects_cache["inversion"].mode = zone
        if "pixel" in self.effects_cache:
            self.effects_cache["pixel"].mode = zone

    def _apply_effect(self, frame):
        key = self.label_to_key.get(self.selected_mode.get())
        intensity = self.intensity.get() / 100.0
        if key is None or intensity == 0:
            return frame

        if key == "pipeline":
            pipe = self._get_pipeline()
            if not pipe.slots:
                return frame
            overlay = pipe.apply(frame)
        else:
            overlay = self._get_single_effect(key).apply(frame)

        alpha = (overlay[:, :, 3].astype("float32") / 255.0) * intensity
        alpha = alpha[:, :, None]
        return (frame.astype("float32") * (1.0 - alpha) + overlay[:, :, :3].astype("float32") * alpha).clip(0, 255).astype("uint8")

    def _start_camera(self):
        try:
            self.engine = VideoEngineThreaded(src=int(self.selected_camera.get()), width=640, height=480, buffer_size=30).start()
        except Exception as error:
            self.engine = None
            self.camera_status.set(f"Error de cámara: {error}")
            self.status_dot.configure(fg="#ee6b72")
            self.camera_button.configure(text="Reintentar cámara")
            return

        self.camera_status.set("Cámara conectada · captura en tiempo real")
        self.status_dot.configure(fg="#61d095")
        self.camera_button.configure(text="Detener cámara")
        self._last_frame_id = None
        self.current_frame = None
        self._reset_all_effects()
        self._fps_frames = 0
        self._fps_started = 0.0

    def _stop_camera(self):
        self._stop_recording()
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self.preview.configure(image="", text="Cámara detenida")
        self.preview_image = None
        self.current_frame = None
        self.camera_status.set("Cámara detenida")
        self.status_dot.configure(fg="#929bad")
        self.fps_text.set("-- FPS · -- × --")
        self.performance_text.set("Buffer: --/30")
        self.camera_button.configure(text="Iniciar cámara")

    def _toggle_camera(self):
        if self.engine is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _change_camera(self, _event=None):
        if self.engine is not None:
            self._stop_camera()
        self._start_camera()

    def _save_snapshot(self):
        if self.current_frame is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=(("Imagen JPEG", "*.jpg"), ("Imagen PNG", "*.png")))
        if path and cv2.imwrite(path, self.current_frame):
            self.camera_status.set(f"Captura guardada: {Path(path).name}")

    def _toggle_recording(self):
        if self.video_writer is not None:
            self._stop_recording()
            return
        if self.current_frame is None:
            return

        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=(("Vídeo MP4", "*.mp4"),))
        if not path:
            return

        h, w = self.current_frame.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        if writer.isOpened():
            self.video_writer = writer
            self._recording_path = path
            self.recording_status.set("● Grabando")
            self.record_button.configure(text="Detener grabación", bg="#7f3039")

    def _stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording_status.set(f"Grabada: {Path(self._recording_path).name}")
            self.record_button.configure(text="Iniciar grabación", bg="#bd4651")

    def _update_preview(self):
        if self.engine is not None:
            frame_id, frame = self.engine.get_latest_frame_with_id()
            if frame is not None and frame_id != self._last_frame_id:
                try:
                    processed = self._apply_effect(frame)
                except Exception:
                    processed = frame

                if self.mirror_enabled.get():
                    processed = cv2.flip(processed, 1)
                
                self.current_frame = processed
                self._last_frame_id = frame_id

                if self.video_writer is not None:
                    self.video_writer.write(processed)

                self._fps_frames += 1
                now = time.perf_counter()
                if self._fps_started == 0.0:
                    self._fps_started = now
                elif now - self._fps_started >= 1.0:
                    self._display_fps = self._fps_frames / (now - self._fps_started)
                    self._fps_frames = 0
                    self._fps_started = now

                self.performance_text.set(f"Buffer: {len(self.engine.get_buffer_as_array())}/30")
                self.fps_text.set(f"{self._display_fps:.0f} FPS · {processed.shape[1]} × {processed.shape[0]}")

                encoded, png_frame = cv2.imencode(".png", processed)
                if encoded:
                    self.preview_image = tk.PhotoImage(data=base64.b64encode(png_frame).decode("ascii"), format="png")
                    self.preview.configure(image=self.preview_image, text="")

        self.root.after(20, self._update_preview)

    def _close(self):
        self._stop_recording()
        if self.engine is not None:
            self.engine.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    VideoInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
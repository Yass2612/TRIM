from __future__ import annotations

import base64
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded

if TYPE_CHECKING:
    from Efecto_Neon import SilhouetteNeonEffect
    from visual_effects import SilhouetteTrailEffect


class VideoInterface:
    BACKGROUND = "#10131c"
    PANEL = "#191e2a"
    MUTED = "#929bad"
    ACCENT = "#7c6cff"

    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de vídeo | Interfaz de prueba")
        self.root.geometry("1180x780")
        self.root.minsize(1000, 700)
        self.root.configure(bg=self.BACKGROUND)

        self.engine = None
        self.preview_image = None
        self.current_frame = None
        self.video_writer = None
        self._last_frame_id = None
        self.selected_effect = tk.StringVar(value="Original")
        self.selected_camera = tk.StringVar(value="0")
        self.mirror_enabled = tk.BooleanVar(value=False)
        self.camera_status = tk.StringVar(value="Conectando con la cámara...")
        self.fps_text = tk.StringVar(value="-- FPS · -- × --")
        self.intensity_text = tk.StringVar(value="50 %")
        self.recording_status = tk.StringVar(value="Grabación detenida")
        self.performance_text = tk.StringVar(value="Buffer: --/30")
        self.trail_effect: SilhouetteTrailEffect | None = None
        self.neon_effect: SilhouetteNeonEffect | None = None
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
        header.pack(fill="x", padx=28, pady=(22, 16))

        tk.Label(
            header,
            text="VIDEO LAB",
            bg=self.BACKGROUND,
            fg="#a99fff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Procesamiento de vídeo en tiempo real",
            bg=self.BACKGROUND,
            fg="#f4f5fa",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            header,
            text="Fase 2 · Interfaz e integración",
            bg=self.BACKGROUND,
            fg=self.MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(3, 0))

        content = tk.Frame(self.root, bg=self.BACKGROUND)
        content.pack(fill="both", expand=True, padx=28, pady=(0, 24))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        preview_panel = tk.Frame(content, bg=self.PANEL, padx=16, pady=16)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        preview_panel.rowconfigure(1, weight=1)
        preview_panel.columnconfigure(0, weight=1)

        preview_header = tk.Frame(preview_panel, bg=self.PANEL)
        preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            preview_header,
            text="VISTA DE CÁMARA",
            bg=self.PANEL,
            fg="#f4f5fa",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        tk.Label(
            preview_header,
            textvariable=self.fps_text,
            bg="#252b39",
            fg="#bcb4ff",
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

        self.preview = tk.Label(
            preview_panel,
            text="Esperando señal de vídeo...",
            bg="#0b0e14",
            fg=self.MUTED,
            font=("Segoe UI", 12),
        )
        self.preview.grid(row=1, column=0, sticky="nsew")

        status_row = tk.Frame(preview_panel, bg=self.PANEL)
        status_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.status_dot = tk.Label(
            status_row, text="●", bg=self.PANEL, fg="#e8b451", font=("Segoe UI", 10)
        )
        self.status_dot.pack(side="left", padx=(0, 7))
        tk.Label(
            status_row,
            textvariable=self.camera_status,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        controls = tk.Frame(content, bg=self.PANEL, width=280, padx=18, pady=18)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.grid_propagate(False)

        tk.Label(
            controls,
            text="CONTROLES",
            bg=self.PANEL,
            fg="#f4f5fa",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            controls,
            text="Selecciona un modo de prueba",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 14))

        tk.Label(
            controls,
            text="CÁMARA",
            bg=self.PANEL,
            fg="#f4f5fa",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        self.camera_selector = ttk.Combobox(
            controls,
            textvariable=self.selected_camera,
            values=tuple(str(index) for index in range(6)),
            state="readonly",
        )
        self.camera_selector.pack(fill="x", pady=(0, 12))
        self.camera_selector.bind("<<ComboboxSelected>>", self._change_camera)

        for effect in ("Original", "Estela", "Neón"):
            tk.Radiobutton(
                controls,
                text=effect,
                value=effect,
                variable=self.selected_effect,
                command=self._select_effect,
                indicatoron=False,
                anchor="w",
                padx=12,
                pady=9,
                bg="#222837",
                fg="#e5e8f0",
                selectcolor=self.ACCENT,
                activebackground="#30374a",
                activeforeground="white",
                font=("Segoe UI", 10),
                relief="flat",
                cursor="hand2",
            ).pack(fill="x", pady=3)

        tk.Label(
            controls,
            text="INTENSIDAD",
            bg=self.PANEL,
            fg="#f4f5fa",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(22, 5))
        self.intensity = tk.DoubleVar(value=50)
        self.intensity_scale = ttk.Scale(
            controls,
            from_=0,
            to=100,
            variable=self.intensity,
            style="Test.Horizontal.TScale",
            command=self._update_intensity_label,
        )
        self.intensity_scale.pack(fill="x")
        self.intensity_scale.configure(state="disabled")
        tk.Label(
            controls,
            textvariable=self.intensity_text,
            bg=self.PANEL,
            fg="#bcb4ff",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="e", pady=(3, 0))

        tk.Checkbutton(
            controls,
            text="Espejo horizontal",
            variable=self.mirror_enabled,
            bg=self.PANEL,
            fg="#e5e8f0",
            activebackground=self.PANEL,
            activeforeground="white",
            selectcolor="#303747",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(12, 0))

        self.effect_note = tk.Label(
            controls,
            text="Modo original · señal sin filtros",
            bg=self.PANEL,
            fg="#bcb4ff",
            font=("Segoe UI", 9),
            wraplength=235,
            justify="left",
        )
        self.effect_note.pack(anchor="w", pady=(12, 8))

        tk.Label(
            controls,
            textvariable=self.performance_text,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        tk.Label(
            controls,
            textvariable=self.recording_status,
            bg=self.PANEL,
            fg="#ee6b72",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 7))

        self.capture_button = tk.Button(
            controls,
            text="Guardar foto",
            command=self._save_snapshot,
            bg="#303747",
            fg="white",
            activebackground="#414b60",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
            cursor="hand2",
        )
        self.capture_button.pack(fill="x", side="bottom", pady=(6, 0))

        self.record_button = tk.Button(
            controls,
            text="Iniciar grabación",
            command=self._toggle_recording,
            bg="#bd4651",
            fg="white",
            activebackground="#a53b45",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
            cursor="hand2",
        )
        self.record_button.pack(fill="x", side="bottom", pady=(0, 6))

        self.camera_button = tk.Button(
            controls,
            text="Detener cámara",
            command=self._toggle_camera,
            bg=self.ACCENT,
            fg="white",
            activebackground="#6a5be0",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            cursor="hand2",
        )
        self.camera_button.pack(fill="x", side="bottom")

    def _start_camera(self):
        try:
            self.engine = VideoEngineThreaded(
                src=int(self.selected_camera.get()),
                width=640,
                height=480,
                buffer_size=30,
            ).start()
        except (ValueError, cv2.error) as error:
            self.engine = None
            self.camera_status.set(f"No se pudo acceder a la cámara: {error}")
            self.status_dot.configure(fg="#ee6b72")
            self.camera_button.configure(text="Reintentar cámara")
            return

        self.camera_status.set("Cámara conectada · captura en tiempo real")
        self.status_dot.configure(fg="#61d095")
        self.camera_button.configure(text="Detener cámara")
        self._last_frame_id = None
        self.current_frame = None
        if self.trail_effect is not None:
            self.trail_effect.reset()
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

    def _select_effect(self):
        effect = self.selected_effect.get()
        if effect in ("Estela", "Neón"):
            try:
                self._get_selected_effect()
            except ImportError as error:
                self._report_effect_error(
                    "No se pudo importar MediaPipe. Instálalo en el entorno de "
                    f"Python de la interfaz con:\n\"{sys.executable}\" -m pip "
                    f"install mediapipe\n\nDetalle: {error}"
                )
                return
            except FileNotFoundError as error:
                self._report_effect_error(str(error))
                return
            except (RuntimeError, ValueError) as error:
                self._report_effect_error(str(error))
                return

        if self.trail_effect is not None:
            self.trail_effect.reset()
        if self.neon_effect is not None:
            self.neon_effect.reset()
        self.intensity_scale.configure(
            state="disabled" if effect == "Original" else "normal"
        )
        if effect == "Original":
            self.effect_note.configure(text="Modo original · señal sin filtros")
        elif effect == "Estela":
            self.effect_note.configure(text="Silueta con estela · intensidad ajustable")
        else:
            self.effect_note.configure(text="Contorno neón · intensidad ajustable")

    def _get_trail_effect(self) -> SilhouetteTrailEffect:
        if self.trail_effect is None:
            from visual_effects import SilhouetteTrailEffect

            self.trail_effect = SilhouetteTrailEffect()
        return self.trail_effect

    def _get_neon_effect(self) -> SilhouetteNeonEffect:
        if self.neon_effect is None:
            from Efecto_Neon import SilhouetteNeonEffect

            self.neon_effect = SilhouetteNeonEffect()
        return self.neon_effect

    def _get_selected_effect(self):
        if self.selected_effect.get() == "Estela":
            return self._get_trail_effect()
        return self._get_neon_effect()

    def _report_effect_error(self, detail):
        self.selected_effect.set("Original")
        self.intensity_scale.configure(state="disabled")
        self.effect_note.configure(text="Efecto no disponible")
        messagebox.showerror(
            "No se pudo cargar el efecto",
            f"No se ha podido iniciar el efecto seleccionado.\n\n{detail}",
        )

    def _update_intensity_label(self, value):
        self.intensity_text.set(f"{float(value):.0f} %")

    def _apply_effect(self, frame):
        effect = self.selected_effect.get()
        intensity = self.intensity.get() / 100.0
        if effect == "Original" or intensity == 0:
            return frame

        if effect in ("Estela", "Neón"):
            overlay = self._get_selected_effect().apply(frame)
            alpha = (overlay[:, :, 3].astype("float32") / 255.0) * intensity
            alpha = alpha[:, :, None]
            return (
                frame.astype("float32") * (1.0 - alpha)
                + overlay[:, :, :3].astype("float32") * alpha
            ).clip(0, 255).astype("uint8")
        return frame

    def _save_snapshot(self):
        if self.current_frame is None:
            messagebox.showwarning("Sin imagen", "Espera a que llegue un fotograma.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar captura",
            defaultextension=".jpg",
            filetypes=(("Imagen JPEG", "*.jpg"), ("Imagen PNG", "*.png")),
        )
        if not path:
            return
        try:
            saved = cv2.imwrite(path, self.current_frame)
        except (cv2.error, OSError) as error:
            messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen: {error}")
            return
        if not saved:
            messagebox.showerror("Error al guardar", "No se pudo guardar la imagen.")
            return
        self.camera_status.set(f"Captura guardada: {Path(path).name}")

    def _toggle_recording(self):
        if self.video_writer is not None:
            self._stop_recording()
            return
        if self.current_frame is None:
            messagebox.showwarning("Sin vídeo", "Espera a que llegue un fotograma.")
            return

        path = filedialog.asksaveasfilename(
            title="Guardar grabación",
            defaultextension=".mp4",
            filetypes=(("Vídeo MP4", "*.mp4"),),
        )
        if not path:
            return

        height, width = self.current_frame.shape[:2]
        try:
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (width, height),
            )
        except (cv2.error, OSError) as error:
            messagebox.showerror("Error al grabar", f"No se pudo crear el vídeo: {error}")
            return
        if not writer.isOpened():
            writer.release()
            messagebox.showerror(
                "Error al grabar",
                "No se pudo crear el vídeo. Prueba otra ubicación o formato.",
            )
            return

        self.video_writer = writer
        self._recording_path = path
        self.recording_status.set("● Grabando")
        self.record_button.configure(text="Detener grabación", bg="#7f3039")

    def _stop_recording(self):
        if self.video_writer is None:
            return
        self.video_writer.release()
        self.video_writer = None
        self.recording_status.set(
            f"Grabación guardada: {Path(self._recording_path).name}"
        )
        self.record_button.configure(text="Iniciar grabación", bg="#bd4651")

    def _update_preview(self):
        if self.engine is not None:
            frame_id, frame = self.engine.get_latest_frame_with_id()
            if frame is not None and frame_id != self._last_frame_id:
                try:
                    processed = self._apply_effect(frame)
                except (RuntimeError, ValueError, cv2.error) as error:
                    self._report_effect_error(str(error))
                    processed = frame
                if self.mirror_enabled.get():
                    processed = cv2.flip(processed, 1)
                self.current_frame = processed
                self._last_frame_id = frame_id

                if self.video_writer is not None:
                    try:
                        self.video_writer.write(processed)
                    except cv2.error as error:
                        self._stop_recording()
                        messagebox.showerror(
                            "Error al grabar",
                            f"Se detuvo la grabación: {error}",
                        )

                self._fps_frames += 1
                now = time.perf_counter()
                if self._fps_started == 0.0:
                    self._fps_started = now
                elif now - self._fps_started >= 1.0:
                    self._display_fps = self._fps_frames / (now - self._fps_started)
                    self._fps_frames = 0
                    self._fps_started = now

                self.performance_text.set(
                    f"Buffer: {len(self.engine.get_buffer_as_array())}/30"
                )
                height, width = processed.shape[:2]
                self.fps_text.set(
                    f"{self._display_fps:.0f} FPS · {width} × {height}"
                )

                encoded, png_frame = cv2.imencode(".png", processed)
                if not encoded:
                    self.camera_status.set("No se pudo convertir el fotograma para mostrarlo")
                    self.status_dot.configure(fg="#ee6b72")
                    self.root.after(20, self._update_preview)
                    return
                self.preview_image = tk.PhotoImage(
                    data=base64.b64encode(png_frame).decode("ascii"),
                    format="png",
                )
                self.preview.configure(image=self.preview_image, text="")

        self.root.after(20, self._update_preview)

    def _close(self):
        self._stop_recording()
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self.root.destroy()


def main():
    root = tk.Tk()
    VideoInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()

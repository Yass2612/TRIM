import base64
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded


class VideoInterface:
    BACKGROUND = "#10131c"
    PANEL = "#191e2a"
    MUTED = "#929bad"
    ACCENT = "#7c6cff"

    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de vídeo | Interfaz de prueba")
        self.root.geometry("1100x720")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.BACKGROUND)

        self.engine = None
        self.preview_image = None
        self.selected_effect = tk.StringVar(value="Original")
        self.camera_status = tk.StringVar(value="Conectando con la cámara...")
        self.fps_text = tk.StringVar(value="-- FPS")
        self._fps_frames = 0
        self._fps_started = 0.0

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

        for effect in ("Original", "Estela", "Neón", "Térmico"):
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
        ttk.Scale(
            controls,
            from_=0,
            to=100,
            variable=self.intensity,
            style="Test.Horizontal.TScale",
            state="disabled",
        ).pack(fill="x")
        tk.Label(
            controls,
            text="Se conectará al integrar los filtros.",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 8),
            wraplength=235,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        self.effect_note = tk.Label(
            controls,
            text="Modo original · señal sin filtros",
            bg=self.PANEL,
            fg="#bcb4ff",
            font=("Segoe UI", 9),
            wraplength=235,
            justify="left",
        )
        self.effect_note.pack(anchor="w", pady=(20, 12))

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
                src=0, width=640, height=480, buffer_size=30
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
        self._fps_frames = 0
        self._fps_started = 0.0

    def _stop_camera(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self.preview.configure(image="", text="Cámara detenida")
        self.preview_image = None
        self.camera_status.set("Cámara detenida")
        self.status_dot.configure(fg="#929bad")
        self.fps_text.set("-- FPS")
        self.camera_button.configure(text="Iniciar cámara")

    def _toggle_camera(self):
        if self.engine is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _select_effect(self):
        effect = self.selected_effect.get()
        if effect == "Original":
            self.effect_note.configure(text="Modo original · señal sin filtros")
        else:
            self.effect_note.configure(
                text=f"{effect} seleccionado · se implementará en la fase 3"
            )

    def _update_preview(self):
        if self.engine is not None:
            frame = self.engine.get_latest_frame()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb_frame.shape[:2]
                ppm = f"P6 {width} {height} 255 ".encode("ascii") + rgb_frame.tobytes()
                self.preview_image = tk.PhotoImage(
                    data=base64.b64encode(ppm), format="PPM"
                )
                self.preview.configure(image=self.preview_image, text="")

                self._fps_frames += 1
                now = cv2.getTickCount() / cv2.getTickFrequency()
                if self._fps_started == 0.0:
                    self._fps_started = now
                elif now - self._fps_started >= 1.0:
                    fps = self._fps_frames / (now - self._fps_started)
                    self.fps_text.set(f"{fps:.0f} FPS")
                    self._fps_frames = 0
                    self._fps_started = now

        self.root.after(20, self._update_preview)

    def _close(self):
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

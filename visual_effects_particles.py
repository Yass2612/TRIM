import cv2
import numpy as np

from Motor_Y_Captura.Motor_y_Captura_v2 import VideoEngineThreaded
from silhouette_tracker import SilhouetteTracker


class SilhouetteParticlesEffect:
    """
    Efecto Chispas / Partículas de Movimiento:
    - Detecta variaciones en el contorno de la silueta entre frames consecutivos.
    - Genera partículas luminosas que caen por gravedad (o flotan) y se apagan.
    """

    def __init__(self, spawn_rate=35, gravity=0.45, max_particles=400):
        self.spawn_rate = spawn_rate
        self.gravity = gravity
        self.max_particles = max_particles
        self.prev_mask = None
        # Cada partícula: [x, y, vx, vy, vida (1.0 a 0.0)]
        self.particles = np.empty((0, 5), dtype=np.float32)
        self.tracker = SilhouetteTracker()

    def reset(self):
        self.tracker.reset()
        self.prev_mask = None
        self.particles = np.empty((0, 5), dtype=np.float32)

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

        # 1. Detectar zonas nuevas de movimiento en el borde de la silueta
        if self.prev_mask is not None and self.prev_mask.shape == mask.shape:
            diff = cv2.absdiff(mask, self.prev_mask)
            ys, xs = np.nonzero(diff > 127)
            if len(xs) > 0:
                count = min(self.spawn_rate, len(xs))
                idx = np.random.choice(len(xs), size=count, replace=False)
                new_x = xs[idx].astype(np.float32)
                new_y = ys[idx].astype(np.float32)
                vx = np.random.uniform(-2.5, 2.5, size=count).astype(np.float32)
                vy = np.random.uniform(-4.0, -0.5, size=count).astype(np.float32)
                life = np.ones(count, dtype=np.float32)
                new_p = np.column_stack([new_x, new_y, vx, vy, life])
                self.particles = np.vstack([self.particles, new_p])[-self.max_particles:]

        self.prev_mask = mask.copy()

        # 2. Actualizar física de las partículas activas
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        alpha_layer = np.zeros((h, w), dtype=np.float32)

        if len(self.particles) > 0:
            self.particles[:, 0] += self.particles[:, 2]  # x += vx
            self.particles[:, 1] += self.particles[:, 3]  # y += vy
            self.particles[:, 3] += self.gravity          # vy += gravedad
            self.particles[:, 4] -= 0.035                 # desgaste de vida

            # Filtrar partículas vivas y dentro de pantalla
            valid = (
                (self.particles[:, 4] > 0)
                & (self.particles[:, 0] >= 0) & (self.particles[:, 0] < w)
                & (self.particles[:, 1] >= 0) & (self.particles[:, 1] < h)
            )
            self.particles = self.particles[valid]

            for x, y, _, _, life in self.particles:
                px, py = int(x), int(y)
                # Color transición: blanco/amarillo -> naranja -> rojo
                color = (
                    int(50 * life),
                    int(220 * life),
                    255,
                )
                radius = max(1, int(3 * life))
                cv2.circle(canvas, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(alpha_layer, (px, py), radius + 1, float(life), -1)

        # Añadir ligero brillo (glow) a las chispas
        glow = cv2.GaussianBlur(canvas, (0, 0), 3.0)
        final_particles = np.clip(canvas + glow * 0.7, 0, 255).astype(np.uint8)
        alpha_u8 = np.clip(alpha_layer * 255.0, 0, 255).astype(np.uint8)

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = final_particles
        out[..., 3] = alpha_u8
        return out


if __name__ == "__main__":
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    effect = SilhouetteParticlesEffect()
    try:
        while True:
            frames = engine.get_buffer_as_array()
            if frames:
                res = effect.apply(frames[-1])
                # Componer sobre el frame original para prueba directa
                a = (res[..., 3:4].astype(np.float32) / 255.0)
                comp = (frames[-1].astype(np.float32) * (1 - a) + res[..., :3].astype(np.float32) * a).astype(np.uint8)
                cv2.imshow("Particulas", comp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()
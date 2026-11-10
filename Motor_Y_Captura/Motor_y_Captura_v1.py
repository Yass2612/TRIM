import cv2
from collections import deque
import time

class VideoEngine:
    def __init__(self, src=0, buffer_size=30):
        """
        Inicializa la captura de cámara y el buffer circular de frames.
        :param src: Índice de la cámara (normalmente 0 para la webcam integrada)
        :param buffer_size: Cantidad de frames a almacenar (20-30 recomendado)
        """
        self.cap = cv2.VideoCapture(src)
        
        # Ajustar resolución para optimizar el uso de CPU (ej. 640x480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Buffer de memoria FIFO (First In, First Out) con tamaño máximo fijo
        self.buffer = deque(maxlen=buffer_size)
        self.is_running = True

        if not self.cap.isOpened():
            raise ValueError("No se pudo acceder a la webcam.")

    def update(self):
        """Lee el frame actual de la webcam y lo almacena en el buffer."""
        ret, frame = self.cap.read()
        if ret:
            # Guardamos una copia limpia en el buffer
            self.buffer.append(frame.copy())
            return frame
        return None

    def get_latest_frame(self):
        """Devuelve el fotograma más reciente (limpio)."""
        if len(self.buffer) > 0:
            return self.buffer[-1]
        return None

    def get_buffer_frames(self):
        """
        Devuelve el historial de fotogramas para el Miembro 2 
        (necesario para el efecto de estela temporal).
        """
        return list(self.buffer)

    def release(self):
        """Libera la cámara y cierra ventanas al terminar."""
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()


# --- PRUEBA DEL MOTOR Y CONTROL DE RENDIMIENTO (FPS) ---
if __name__ == "__main__":
    engine = VideoEngine(src=0, buffer_size=30)
    
    prev_time = time.time()
    
    print("Iniciando motor de captura. Presiona 'q' para salir.")
    
    while True:
        frame = engine.update()
        
        if frame is None:
            break
            
        # Cálculo simple de FPS en tiempo real
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Superponer los FPS en pantalla para analizar el rendimiento (Tarea de Diciembre)
        cv2.putText(frame, f"FPS: {int(fps)} | Buffer: {len(engine.buffer)} frames", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar la señal limpia
        cv2.imshow("Motor y Captura (Miembro 1)", frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.release()
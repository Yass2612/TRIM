import cv2
import numpy as np
from collections import deque
import threading
import time

class VideoEngineThreaded:
    def __init__(self, src=0, width=640, height=480, buffer_size=30):
        """
        Motor de captura de vídeo multi-hilo optimizado para tiempo real.
        
        :param src: Índice de la webcam (0 por defecto).
        :param width: Ancho de resolución.
        :param height: Alto de resolución.
        :param buffer_size: Tamaño del historial de frames para el efecto estela.
        """
        self.cap = cv2.VideoCapture(src)
        
        # Optimización de hardware/resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Desactivar buffer interno de OpenCV para reducir el lag / latencia
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ValueError("[ERROR] No se pudo acceder a la webcam.")

        # Buffer circular FIFO para almacenar fotogramas
        self.buffer = deque(maxlen=buffer_size)
        
        # Variables de control de hilos e hilos seguros (Thread Lock)
        self.lock = threading.Lock()
        self.running = False
        self.current_frame = None
        self.frame_id = 0
        
        # Lectura inicial del primer frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.buffer.append(frame.copy())
            self.frame_id += 1

    def start(self):
        """Inicia el hilo secundario de captura continua."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        """Bucle interno que se ejecuta en el hilo secundario."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[AVISO] Pérdida de frame de la cámara.")
                continue

            with self.lock:
                self.current_frame = frame
                self.buffer.append(frame.copy())
                self.frame_id += 1

            # Pequeña pausa para no saturar 100% el hilo del procesador
            time.sleep(0.005)

    def get_latest_frame(self):
        """Devuelve el frame actual limpio (para mostrar en la GUI del Miembro 3)."""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_latest_frame_with_id(self):
        """Devuelve el identificador y una copia del frame actual."""
        with self.lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            return self.frame_id, frame

    def get_buffer_as_array(self):
        """
        Devuelve el historial de frames estructurado para el Miembro 2.
        Facilita el cálculo matricial rápido con NumPy para los filtros de estela.
        """
        with self.lock:
            if len(self.buffer) == 0:
                return []
            return list(self.buffer)

    def stop(self):
        """Detiene el hilo de lectura y libera los recursos de hardware."""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        print("[INFO] Motor de captura detenido correctamente.")


# --- EJEMPLO DE PRUEBA Y CONEXIÓN PARA TUS COMPAÑEROS ---
if __name__ == "__main__":
    # Inicializamos el motor con resolución reducida para CPU (640x480)
    engine = VideoEngineThreaded(src=0, width=640, height=480, buffer_size=30).start()
    
    fps_counter = 0
    start_time = time.time()
    fps_display = "0 FPS"

    print("--- MOTOR DE CAPTURA EN EJECUCIÓN ---")
    print("Presiona 'q' para salir.")

    try:
        while True:
            # 1. Obtener frame limpio
            frame = engine.get_latest_frame()
            if frame is None:
                continue

            # 2. Cálculo de rendimiento (FPS)
            fps_counter += 1
            if time.time() - start_time >= 1.0:
                fps_display = f"FPS: {fps_counter}"
                fps_counter = 0
                start_time = time.time()

            # 3. Muestra de estadísticas en la ventana de prueba
            buffer_len = len(engine.get_buffer_as_array())
            info_text = f"{fps_display} | Buffer cargado: {buffer_len}/30 frames"
            
            cv2.putText(frame, info_text, (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 4. Renderizado directo OpenCV (solo para testeo interno)
            cv2.imshow("Test Motor - Miembro 1 (Yassir)", frame)

            # Salir pulsando la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        engine.stop()
        cv2.destroyAllWindows()
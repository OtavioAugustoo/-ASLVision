from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import joblib
import numpy as np

# Inicializar Flask
app = Flask(__name__)

# Carregar modelo treinado
modelo = joblib.load("models/random_forest_model.pkl")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Abrir webcam
cap = cv2.VideoCapture(0)

def gerar_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        letra_predita = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pontos = []
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    pontos.append(lm.x - wrist.x)
                    pontos.append(lm.y - wrist.y)

                if len(pontos) == 42:
                    entrada = np.array(pontos).reshape(1, -1)
                    letra_predita = modelo.predict(entrada)[0]

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar letra predita
        if letra_predita:
            cv2.putText(frame, f"Letra: {letra_predita}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

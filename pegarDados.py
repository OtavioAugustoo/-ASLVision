import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Pasta para salvar o CSV
os.makedirs("data", exist_ok=True)

def coletar_letra(letra, num_amostras=300):
    dados = []
    cap = cv2.VideoCapture(0)
    print(f"➡️ Posicione sua mão para a letra '{letra}'. Capturando {num_amostras} amostras...")

    amostras_coletadas = 0
    while amostras_coletadas < num_amostras:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pontos = []
                wrist = hand_landmarks.landmark[0]
                for lm in hand_landmarks.landmark:
                    # Normalizar em relação ao pulso (landmark 0)
                    pontos.append(lm.x - wrist.x)
                    pontos.append(lm.y - wrist.y)

                dados.append(pontos + [letra])
                amostras_coletadas += 1

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Letra: {letra} | Amostras: {amostras_coletadas}/{num_amostras}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Coleta ASL", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salvar no CSV
    colunas = [f'x{i}' if i % 2 == 0 else f'y{i//2}' for i in range(42)] + ['label']
    df = pd.DataFrame(dados, columns=colunas)

    arquivo_csv = "data/asl_dataset.csv"
    if os.path.exists(arquivo_csv):
        df.to_csv(arquivo_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(arquivo_csv, index=False)

    print(f" Coleta da letra '{letra}' finalizada.")

# Coletar várias letras (edite conforme necessidade)
if __name__ == "__main__":
    letras = input("Digite as letras que deseja coletar (ex: abcdef): ").strip().lower()
    for letra in letras:
        coletar_letra(letra)

import argparse
import cv2
from deepface import DeepFace

EMOTION_COLORS = {
    "happy":    (0, 255, 0),
    "sad":      (255, 0, 0),
    "angry":    (0, 0, 255),
    "fear":     (128, 0, 128),
    "surprise": (0, 255, 255),
    "disgust":  (0, 128, 0),
    "neutral":  (200, 200, 200),
}

EMOTION_FR = {
    "happy":    "Joie",
    "sad":      "Tristesse",
    "angry":    "Colere",
    "fear":     "Peur",
    "surprise": "Surprise",
    "disgust":  "Degout",
    "neutral":  "Neutre",
}

def draw_face(frame, region, dominant, score):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    color = EMOTION_COLORS.get(dominant, (255, 255, 255))
    label = f"{EMOTION_FR.get(dominant, dominant)} ({score:.0f}%)"
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Fond du label
    label_y = y - 10 if y > 30 else y + h + 25
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, label_y - th - 5), (x + tw + 6, label_y + 4), color, -1)
    cv2.putText(frame, label, (x + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

def run_emotion_detection(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la source vidéo.")
        return


    # Configuration de la fenêtre
    window_name = "Emotion Detection - Ekman"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)  
    cv2.resizeWindow(window_name, 640, 480)
          # redimensionnable
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # au premier plan

    # Supprimer la barre de titre / décoration (plein écran sans barre)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Détection démarrée. Appuyer sur 'q' pour quitter.")
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True
            )
            # DeepFace retourne un dict si 1 visage, une liste si plusieurs
            if isinstance(results, dict):
                results = [results]

        except Exception:
            results = []

        for face in results:
            dominant = face["dominant_emotion"]
            score = face["emotion"][dominant]
            draw_face(frame, face["region"], dominant, score)

        # Compteur visages
        cv2.putText(
            frame, f"Visages : {len(results)}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("Emotion Detection - Ekman", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection d'émotions en temps réel (Ekman)")
    
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--webcam",
        type=int,
        default=0,
        metavar="ID",
        help="Index de la webcam (défaut: 0)"
    )
    source_group.add_argument(
        "--video",
        type=str,
        metavar="FICHIER",
        help="Chemin vers un fichier vidéo (mp4, avi...)"
    )

    args = parser.parse_args()

    if args.video:
        run_emotion_detection(source=args.video)
    else:
        run_emotion_detection(source=args.webcam)



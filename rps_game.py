"""
Hand Gesture Rock-Paper-Scissors  (You vs Computer)
Controls:  SPACE = start a round   |   Q = quit
Needs:     pip install opencv-python mediapipe
"""

import random
import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Landmark ids: fingertip and the joint below it (index, middle, ring, pinky)
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]

WINS = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}


def get_gesture(lm):
    """Return Rock / Paper / Scissors from hand landmarks (None if unclear)."""
    up = [lm[t].y < lm[p].y for t, p in zip(TIPS, PIPS)]
    if not any(up):
        return "Rock"
    if all(up):
        return "Paper"
    if up[0] and up[1] and not up[2] and not up[3]:
        return "Scissors"
    return None


def decide(player, comp):
    if player == comp:
        return "Draw!"
    return "You Win!" if (player, comp) in WINS else "Computer Wins!"


def put(frame, text, pos, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    cap = cv2.VideoCapture(0)  # if camera doesn't open, try 1
    if not cap.isOpened():
        print("Camera open aagala. VideoCapture(0) ah 1 nu maathi try pannunga.")
        return

    state = "idle"  # idle -> countdown -> result
    start = 0.0
    player = comp = result = ""
    score = {"you": 0, "comp": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w = frame.shape[:2]

        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hl.landmark)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # --- scoreboard ---
        put(frame, f"You {score['you']}  -  {score['comp']} Computer", (20, 40), 1.0, (0, 255, 255))
        put(frame, f"Seeing: {gesture or '...'}", (20, h - 20), 0.8, (200, 255, 200))

        if state == "idle":
            put(frame, "Press SPACE to play", (w // 2 - 200, h // 2), 1.2, (255, 255, 0))
            if key == 32:
                state, start = "countdown", time.time()

        elif state == "countdown":
            left = 3 - int(time.time() - start)
            if left > 0:
                put(frame, str(left), (w // 2 - 30, h // 2), 4.0, (0, 165, 255), 6)
                put(frame, "Get your hand ready!", (w // 2 - 200, h // 2 + 80), 1.0)
            else:
                if gesture is None:
                    result, player, comp = "No hand / unclear - try again", "", ""
                else:
                    player = gesture
                    comp = random.choice(["Rock", "Paper", "Scissors"])
                    result = decide(player, comp)
                    if result == "You Win!":
                        score["you"] += 1
                    elif result == "Computer Wins!":
                        score["comp"] += 1
                state = "result"

        elif state == "result":
            if player:
                put(frame, f"You: {player}", (20, 100), 1.0)
                put(frame, f"Computer: {comp}", (20, 145), 1.0)
            colour = (0, 255, 0) if result == "You Win!" else (0, 0, 255) if result == "Computer Wins!" else (255, 255, 255)
            put(frame, result, (w // 2 - 200, h // 2), 1.5, colour, 3)
            put(frame, "SPACE = next round", (w // 2 - 150, h // 2 + 60), 0.9, (255, 255, 0))
            if key == 32:
                state, start = "countdown", time.time()

        cv2.imshow("Rock Paper Scissors - AI", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

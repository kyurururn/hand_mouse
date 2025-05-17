import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque
import threading

# PyAutoGUIの待機時間を無効化して高速化
pyautogui.PAUSE = 0
# PyAutoGUIのフェイルセーフを無効化（オプション、注意して使用）
pyautogui.FAILSAFE = False

# MediaPipe設定の最適化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3  # トラッキング閾値を下げて追跡性能を維持しながら処理を軽量化
)

# カメラ設定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小化して遅延を減らす

# 画面サイズ取得
screen_width, screen_height = pyautogui.size()

# 定数定義
CLICK_THRESHOLD = 25
SCROLL_THRESHOLD = 25
CAMERA_MARGIN = 100
SAFE_MARGIN = 5
CLICK_INTERVAL = 0.2
DOUBLE_CLICK_INTERVAL = 0.5
SCROLL_SENSITIVITY = 2
SMOOTHING_FACTOR = 0.8  # スムージング係数を少し上げて反応速度向上

# 線形補完の設定
INTERPOLATION_STEPS = 1  # 補完するステップ数（多いほど滑らかだが、レイテンシが増加）
INTERPOLATION_DELAY = 0.008  # 補完ステップ間の遅延（秒）

# バッファサイズを小さくして反応速度向上
BUFFER_SIZE = 3
smooth_x_buffer = deque(maxlen=BUFFER_SIZE)
smooth_y_buffer = deque(maxlen=BUFFER_SIZE)

# 状態管理変数
mouse_down = False
last_click_time = 0
last_double_click_time = 0
prev_mouse_x, prev_mouse_y = screen_width // 2, screen_height // 2
target_mouse_x, target_mouse_y = prev_mouse_x, prev_mouse_y  # 線形補完のための目標座標
right_hand_previous_y = None  # 変数名を変更（左手→右手）

# 前回の手の位置をキャッシュ
last_hand_positions = {}

# フレームスキップ用カウンター（必要に応じて処理を間引く）
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 1  # 全フレーム処理する場合は1、間引く場合は2以上

# マウス操作をスレッド化するためのキュー
mouse_action_queue = deque(maxlen=30)  # キューサイズを拡大して線形補完用のステップを確保

def linear_interpolation(start_x, start_y, end_x, end_y, steps):
    """2点間の線形補完を行い、中間点の配列を返す"""
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t
        points.append((int(x), int(y)))
    return points

def perform_mouse_actions():
    """マウス操作を別スレッドで実行するための関数"""
    global mouse_down, prev_mouse_x, prev_mouse_y
    
    while cap.isOpened():
        if mouse_action_queue:
            action = mouse_action_queue.popleft()
            action_type = action.get('type')
            
            if action_type == 'move':
                # 通常の移動
                pyautogui.moveTo(action['x'], action['y'])
                prev_mouse_x, prev_mouse_y = action['x'], action['y']
            elif action_type == 'moveInterpolated':
                # 線形補完による移動
                start_x, start_y = action['start_x'], action['start_y']
                end_x, end_y = action['end_x'], action['end_y']
                
                # 距離が大きい場合のみ補完（小さな動きは直接移動）
                distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                if distance > 20:  # 20ピクセル以上の移動の場合に補完
                    # 距離に応じて補完ステップ数を動的に調整（オプション）
                    steps = min(max(int(distance / 10), 2), INTERPOLATION_STEPS)
                    
                    points = linear_interpolation(start_x, start_y, end_x, end_y, steps)
                    for point_x, point_y in points:
                        pyautogui.moveTo(point_x, point_y)
                        prev_mouse_x, prev_mouse_y = point_x, point_y
                        time.sleep(INTERPOLATION_DELAY)
                else:
                    # 小さな移動は直接実行
                    pyautogui.moveTo(end_x, end_y)
                    prev_mouse_x, prev_mouse_y = end_x, end_y
            elif action_type == 'mouseDown':
                pyautogui.mouseDown()
                mouse_down = True
            elif action_type == 'mouseUp':
                pyautogui.mouseUp()
                mouse_down = False
            elif action_type == 'doubleClick':
                pyautogui.doubleClick()
            elif action_type == 'scroll':
                pyautogui.scroll(action['amount'])
        
        time.sleep(0.001)  # CPU負荷軽減のための短い待機

# マウス操作スレッド開始
mouse_thread = threading.Thread(target=perform_mouse_actions)
mouse_thread.daemon = True
mouse_thread.start()

try:
    while cap.isOpened():
        # フレーム取得
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームスキップ処理（必要に応じて）
        frame_counter += 1
        if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
            continue
        
        # 画像の縮小処理（処理を軽量化）
        # frame = cv2.resize(frame, (320, 240))  # 解像度を落とす場合はコメント解除
        
        # BGR -> RGB変換と反転
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        # MediaPipeに画像を渡す際、コピーを防止
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # MediaPipeから手の左右情報を取得
                hand_type = handedness.classification[0].label  # 'Left' または 'Right' が返される
                
                # フリップした画像に対応するため、左右を反転
                # カメラ画像を水平方向に反転しているため、MediaPipeが「Left」と検出した場合は実際は右手、
                # 「Right」と検出した場合は実際は左手
                actual_hand_type = 'Left' if hand_type == 'Left' else 'Right'
                
                # 必要なランドマークのみ抽出して計算を軽量化
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                h, w, _ = frame.shape

                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(image, str(id), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h), thumb_tip.z)
                index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h), index_finger_tip.z)
                
                # 親指と人差し指の中間点を計算
                avg_x = (thumb_tip_coords[0] + index_finger_tip_coords[0]) // 2
                avg_y = (thumb_tip_coords[1] + index_finger_tip_coords[1]) // 2
                
                # 実際の手の種類に応じた処理
                if actual_hand_type == 'Right':  # 右手: マウス移動と左クリック操作
                    # 画面座標への変換を効率化
                    mouse_x = np.clip((avg_x - CAMERA_MARGIN) * screen_width / (w - 2 * CAMERA_MARGIN), 
                                    SAFE_MARGIN, screen_width - SAFE_MARGIN)
                    mouse_y = np.clip((avg_y - CAMERA_MARGIN) * screen_height / (h - 2 * CAMERA_MARGIN), 
                                    SAFE_MARGIN, screen_height - SAFE_MARGIN)
                    
                    # スムージングバッファ更新
                    smooth_x_buffer.append(mouse_x)
                    smooth_y_buffer.append(mouse_y)
                    
                    # バッファが十分なデータを持つ場合のみスムージング適用
                    if len(smooth_x_buffer) > 0:
                        smooth_mouse_x = sum(smooth_x_buffer) / len(smooth_x_buffer)
                        smooth_mouse_y = sum(smooth_y_buffer) / len(smooth_y_buffer)
                        
                        # 指数移動平均フィルタ（より効率的なスムージング）
                        final_mouse_x = prev_mouse_x + (smooth_mouse_x - prev_mouse_x) * SMOOTHING_FACTOR
                        final_mouse_y = prev_mouse_y + (smooth_mouse_y - prev_mouse_y) * SMOOTHING_FACTOR
                        
                        # 線形補完を使用したマウス移動
                        mouse_action_queue.append({
                            'type': 'moveInterpolated',
                            'start_x': int(prev_mouse_x),
                            'start_y': int(prev_mouse_y),
                            'end_x': int(final_mouse_x),
                            'end_y': int(final_mouse_y)
                        })
                        
                        # この時点では実際の座標更新はしない（moveInterpolated内で更新される）
                        target_mouse_x, target_mouse_y = final_mouse_x, final_mouse_y
                    
                    # 親指と人差し指の距離計算を最適化（2D距離のみを使用）
                    distance = math.sqrt(
                        (index_finger_tip_coords[0] - thumb_tip_coords[0]) ** 2 +
                        (index_finger_tip_coords[1] - thumb_tip_coords[1]) ** 2
                    )
                    
                    current_time = time.time()
                    
                    # クリック検出と処理
                    if distance < CLICK_THRESHOLD:
                        if not mouse_down and (current_time - last_click_time > CLICK_INTERVAL):
                            mouse_action_queue.append({'type': 'mouseDown'})
                            last_click_time = current_time
                            
                            # ダブルクリック処理
                            if (current_time - last_double_click_time) < DOUBLE_CLICK_INTERVAL:
                                mouse_action_queue.append({'type': 'doubleClick'})
                                last_double_click_time = 0
                            else:
                                last_double_click_time = current_time
                    else:
                        if mouse_down:
                            mouse_action_queue.append({'type': 'mouseUp'})
                
                elif actual_hand_type == 'Left':  # 左手: スクロール操作
                    # 親指と人差し指の距離（2D距離のみを使用）
                    distance = math.sqrt(
                        (index_finger_tip_coords[0] - thumb_tip_coords[0]) ** 2 +
                        (index_finger_tip_coords[1] - thumb_tip_coords[1]) ** 2
                    )
                    
                    if distance < SCROLL_THRESHOLD:
                        if right_hand_previous_y is not None:
                            # スクロール量計算を最適化
                            scroll_amount = int(3 * (index_finger_tip_coords[1] - right_hand_previous_y) * SCROLL_SENSITIVITY)
                            if abs(scroll_amount) > 0:  # 小さすぎる動きを無視
                                mouse_action_queue.append({
                                    'type': 'scroll',
                                    'amount': scroll_amount
                                })
                        right_hand_previous_y = index_finger_tip_coords[1]
                    else:
                        right_hand_previous_y = None
                
                # 手の位置をキャッシュ
                last_hand_positions[actual_hand_type] = (avg_x, avg_y)
                
                # デバッグ用に手のタイプを表示
                cv2.putText(image, f"{actual_hand_type} Hand", 
                           (avg_x - 30, avg_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            right_hand_previous_y = None
        
        # デバッグ表示（不要な場合はコメントアウト）
        # cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        #     break

finally:
    # リソース解放
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
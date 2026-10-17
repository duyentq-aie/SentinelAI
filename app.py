import cv2
import numpy as np
import threading
import uvicorn
import os
import time
import shutil
from collections import defaultdict 
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

# ==========================================================
# 0. INITIALIZE DIRECTORIES & FASTAPI
# ==========================================================
os.makedirs("static/violations", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
violation_counters = {}

# ==========================================================
# 1. LOAD AI MODELS (TRACKING EDITION)
# ==========================================================
print("[*] Loading AI Models (Anti-Spam Tracking Edition)...")
helmet_model = YOLO("helmet_best.pt")
standard_coco_model = YOLO("yolo11s.pt")  
plate_model = YOLO("plate_best.pt")

current_source = "test_video.mp4"
cap = cv2.VideoCapture(current_source)
lock = threading.Lock()

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
violation_records = []  
captured_violator_ids = set() 

# ==========================================================
# 2. COMPUTER VISION UTILITIES
# ==========================================================
def draw_sleek_box(img, x1, y1, x2, y2, label, color, thickness=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, text_color, text_thickness)
    return img

def detect_traffic_light_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    lower_green, upper_green = np.array([40, 50, 50]), np.array([90, 255, 255])
    
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    if cv2.countNonZero(mask_red) > cv2.countNonZero(mask_green) and cv2.countNonZero(mask_red) > 5: 
        return "RED"
    elif cv2.countNonZero(mask_green) > cv2.countNonZero(mask_red) and cv2.countNonZero(mask_green) > 5: 
        return "GREEN"
    return "UNKNOWN"

# ==========================================================
# 3. GENERATOR FUNCTION (MJPEG STREAMING)
# ==========================================================
def generate_frames():
    global cap, current_source, violation_records, captured_violator_ids
    
    # Red Light Zone Polygon
    pts = np.array([[250, 250], [750, 250], [820, 380], [150, 380]], np.int32)
    violation_zone = pts.reshape((-1, 1, 2))

    # =========================================================================
    # WRONG LANE
    # =========================================================================
    bike_lane_zone = None 
    # bike_lane_zone = np.array([[750, 450], [1280, 450], [1280, 720], [650, 720]], np.int32).reshape((-1, 1, 2))
    # =========================================================================

    track_history = defaultdict(lambda: [])
    
    # --- Biến đếm thời gian cho FPS ---
    prev_frame_time = 0 

    while True:
        with lock:
            if cap is None or not cap.isOpened(): continue
            success, frame = cap.read()
            if not success:
                if isinstance(current_source, str) and not current_source.startswith("http"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        frame = cv2.resize(frame, (1280, 720))
        boxes_to_process = [] 
        current_light_state = "UNKNOWN"

        if bike_lane_zone is not None:
            cv2.polylines(frame, [bike_lane_zone], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "FAKE BIKE LANE", (750, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # --- STAGE 1: COCO VEHICLES & TRAFFIC LIGHTS ---
        coco_results = standard_coco_model.track(frame, persist=True, conf=0.35, verbose=False)
        
        if coco_results[0].boxes.id is not None:
            boxes = coco_results[0].boxes.xyxy.cpu()
            track_ids = coco_results[0].boxes.id.int().cpu().tolist()
            clss = coco_results[0].boxes.cls.cpu().tolist()
            
            for box, t_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                name = standard_coco_model.names[int(cls)]
                
                if name == 'traffic light':
                    tl_crop = frame[y1:y2, x1:x2]
                    if tl_crop.size > 0:
                        current_light_state = detect_traffic_light_color(tl_crop)
                        color = (0, 0, 255) if current_light_state == "RED" else (0, 255, 0)
                        frame = draw_sleek_box(frame, x1, y1, x2, y2, f"LIGHT: {current_light_state}", color, 2)
                        
                elif name in vehicle_classes and name != "motorcycle":
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track = track_history[t_id]
                    track.append((cx, cy))
                    if len(track) > 30: track.pop(0)

                    is_violator = False
                    violation_reason = ""

                    # =========================================================================
                    # WRONG-WAY DETECTION LOGIC:
                    # =========================================================================
                    if len(track) >= 15:
                        old_cx, old_cy = track[0]
                        dy = cy - old_cy
                        
                        if dy < -30: 
                            is_violator = True
                            violation_reason = "WRONG-WAY"

                        # if dy > 30: 
                        #     is_violator = True
                        #     violation_reason = "WRONG-WAY"
                    # =========================================================================

                    # Logic 2: WRONG-LANE DETECTION (ONLY FOR CARS, TRUCKS & BUSES)
                    if bike_lane_zone is not None:
                        bottom_center = (int((x1 + x2) / 2), int(y2))
                        if name in ['car', 'truck', 'bus']:
                            if cv2.pointPolygonTest(bike_lane_zone, bottom_center, False) >= 0:
                                is_violator = True
                                violation_reason = "WRONG LANE & WAY" if violation_reason == "WRONG-WAY" else "WRONG LANE"

                    boxes_to_process.append({
                        "id": f"C_{t_id}", "coords": (x1, y1, x2, y2), 
                        "type": name, "violator": is_violator, "reason": violation_reason
                    })

        # --- STAGE 1.5: HELMETS, MOTORCYCLES & OVERLOAD ---
        helmet_results = helmet_model.track(frame, persist=True, conf=0.55, verbose=False)
        motorcycles, heads = [], []
        
        if helmet_results[0].boxes.id is not None:
            boxes = helmet_results[0].boxes.xyxy.cpu()
            track_ids = helmet_results[0].boxes.id.int().cpu().tolist()
            clss = helmet_results[0].boxes.cls.cpu().tolist()
            
            for box, t_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                name = helmet_model.names[int(cls)]
                
                if name == "motorcycle": 
                    motorcycles.append((x1, y1, x2, y2, t_id))
                elif name in ["helmet", "no-helmet"]: 
                    heads.append((x1, y1, x2, y2, name))

        for mx1, my1, mx2, my2, m_id in motorcycles:
            is_violator = False
            violation_reason = ""
            passenger_count = 0
            associated_heads = []
            current_frame_no_helmet = False 
            
            for hx1, hy1, hx2, hy2, h_type in heads:
                cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                    passenger_count += 1
                    associated_heads.append((hx1, hy1, hx2, hy2, h_type))
                    if h_type == "no-helmet":
                        current_frame_no_helmet = True 

            if m_id not in violation_counters:
                violation_counters[m_id] = 0
                
            if current_frame_no_helmet:
                violation_counters[m_id] += 1 
            else:
                violation_counters[m_id] = max(0, violation_counters[m_id] - 1)

            if violation_counters[m_id] >= 5:
                is_violator = True 
                violation_reason = "NO-HELMET"

            for hx1, hy1, hx2, hy2, h_type in associated_heads:
                head_color = (255, 144, 30) if h_type == "helmet" else (0, 0, 255)
                frame = draw_sleek_box(frame, hx1, hy1, hx2, hy2, h_type.upper(), head_color)

            if passenger_count > 2:
                is_violator = True
                violation_reason = "OVERLOAD & NO-HELMET" if violation_reason == "NO-HELMET" else "OVERLOAD"

            boxes_to_process.append({
                "id": f"M_{m_id}", "coords": (mx1, my1, mx2, my2), 
                "type": "motorcycle", "violator": is_violator, "reason": violation_reason
            })

        # --- RED LIGHT VIOLATION CHECK ---
        zone_color = (0, 0, 255) if current_light_state == "RED" else (0, 255, 0)
        cv2.polylines(frame, [violation_zone], isClosed=True, color=zone_color, thickness=2)
        cv2.putText(frame, f"TRAFFIC STATUS: {current_light_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)

        for idx, vehicle in enumerate(boxes_to_process):
            vx1, vy1, vx2, vy2 = vehicle["coords"]
            v_id = vehicle["id"]
            
            if current_light_state == "RED":
                bottom_center = (int((vx1 + vx2) / 2), int(vy2))
                if cv2.pointPolygonTest(violation_zone, bottom_center, False) >= 0:
                    boxes_to_process[idx]["violator"] = True
                    boxes_to_process[idx]["reason"] = "RED-LIGHT RUNNER"
                    frame = draw_sleek_box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} RED-LIGHT", (0, 0, 255), 2)
            
            if not boxes_to_process[idx]["violator"]:
                box_color = (0, 165, 255) if vehicle["type"] == "motorcycle" else (255, 255, 0)
                frame = draw_sleek_box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} {vehicle['type'].upper()}", box_color)
            elif boxes_to_process[idx]["reason"] != "RED-LIGHT RUNNER":
                frame = draw_sleek_box(frame, vx1, vy1, vx2, vy2, f"ID:{v_id} {boxes_to_process[idx]['reason']}", (0, 0, 255), 2)

        # --- STAGE 2: LICENSE PLATES (ANTI-SPAM CAPTURE) ---
        for vehicle in boxes_to_process:
            vx1, vy1, vx2, vy2 = vehicle["coords"]
            is_violator = vehicle["violator"]
            v_id = vehicle["id"]
            
            if is_violator:
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(frame.shape[1], vx2), min(frame.shape[0], vy2)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                if vehicle_crop.size > 0:
                    plate_results = plate_model(vehicle_crop, stream=False, conf=0.35, verbose=False)
                    
                    for p_box in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        rx1, ry1 = vx1 + px1, vy1 + py1
                        rx2, ry2 = vx1 + px2, vy1 + py2
                        
                        frame = draw_sleek_box(frame, rx1, ry1, rx2, ry2, "PLATE", (0, 255, 0), thickness=2)
                        
                        if v_id not in captured_violator_ids:
                            margin = 10
                            crop_y1, crop_y2 = max(0, ry1-margin), min(frame.shape[0], ry2+margin)
                            crop_x1, crop_x2 = max(0, rx1-margin), min(frame.shape[1], rx2+margin)
                            plate_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            if plate_img.size > 0:
                                filename = f"v_{int(time.time())}_{v_id}.jpg"
                                filepath = f"static/violations/{filename}"
                                cv2.imwrite(filepath, plate_img)
                                violation_records.insert(0, f"/{filepath}")
                                
                                captured_violator_ids.add(v_id) 
                                
                                if len(violation_records) > 12: 
                                    violation_records.pop()

        # --- CHÈN CHỈ SỐ FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================================
# 4. API ENDPOINTS
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/update_camera")
async def update_camera(new_source: str = Form(...)):
    global cap, current_source, captured_violator_ids
    with lock:
        if cap is not None: cap.release()
        if new_source.strip().isdigit(): 
            current_source = int(new_source.strip())
        else: 
            current_source = new_source.strip()
        cap = cv2.VideoCapture(current_source)
        captured_violator_ids.clear()
    return {"status": "success", "source": str(current_source)}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global cap, current_source, captured_violator_ids
    file_location = f"static/uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    with lock:
        if cap is not None: cap.release()
        current_source = file_location
        cap = cv2.VideoCapture(current_source)
        captured_violator_ids.clear()
        
    return {"status": "success", "filename": file.filename}

@app.get("/api/violations")
async def get_violations():
    return {"images": violation_records}

if __name__ == "__main__":
    print("[*] Starting SentinelAI FastApi Server...")
    # http://127.0.0.1:8000/
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
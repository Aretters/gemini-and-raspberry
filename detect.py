# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
整合Gemini AI的YOLOv5物體檢測系統
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import torch
import cv2
import json # 新增: 處理 JSON 需要
import requests # 新增: 發送 HTTP 請求需要
import datetime # 新增: 獲取時間戳需要

# Gemini AI整合
try:
    import google.generativeai as genai
    GEMINI_ENABLED = True
except ImportError:
    GEMINI_ENABLED = False
    print("警告: 未找到google-generativeai庫，Gemini功能將禁用")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目錄
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def init_gemini(api_key=None):
    """初始化Gemini AI模型"""
    if not GEMINI_ENABLED:
        return None

    try:
        if api_key:
            genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-001")  # 注意這裡需使用完整名稱
        return model
    except Exception as e:
        print(f"Gemini初始化失敗: {e}")
        return None


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路徑
        source=ROOT / 'data/images',  # 數據源
        data=ROOT / 'data/coco128.yaml',  # 數據集yaml
        imgsz=(640, 640),  # 推理尺寸
        conf_thres=0.25,  # 置信度閾值
        iou_thres=0.45,  # NMS IOU閾值
        max_det=1000,  # 最大檢測數
        device='',  # 設備
        view_img=False,  # 顯示結果
        save_txt=False,  # 保存結果到txt
        save_conf=False,  # 保存置信度
        save_crop=False,  # 保存裁剪框
        nosave=False,  # 不保存圖像/視頻
        classes=None,  # 過濾類別
        agnostic_nms=False,  # 類別無關NMS
        augment=False,  # 增強推理
        visualize=False,  # 可視化特徵
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 保存結果路徑
        name='exp',  # 結果名稱
        exist_ok=False,  # 允許覆蓋
        line_thickness=3,  # 邊框粗細
        hide_labels=False,  # 隱藏標籤
        hide_conf=False,  # 隱藏置信度
        half=False,  # FP16半精度推理
        dnn=False,  # 使用OpenCV DNN
        vid_stride=1,  # 視頻幀步長
        gemini_interval=2,  # Gemini響應間隔
        disable_gemini=False,  # 禁用Gemini
        gemini_api_key=None,  # Gemini API密鑰
):
    # ... (run 函數開頭的初始化代碼) ...
    # === 新增：定義樹莓派的 HTTP 接收 URL ===
    # 請將這裡的 IP 位址和 Port 替換為您樹莓派實際的設定
    raspberry_pi_ip = "192.168.2.227" # 例如: "192.168.1.105"
    raspberry_pi_port = 5000 # 例如: 5000 (如果樹莓派上 Flask 跑在 Port 5000)
    raspberry_pi_command_url = f"http://{raspberry_pi_ip}:{raspberry_pi_port}/command" # /command 是假設樹莓派上的接收路徑
    # 初始化Gemini
    gemini_model = None
    if GEMINI_ENABLED and not disable_gemini:
        gemini_model = init_gemini(gemini_api_key)
        if not gemini_model:
            print("Gemini功能不可用，將繼續執行無Gemini模式")
    
    # 初始化檢測變量
    last_detected = []
    last_gemini_time = 0
    
    # 原始YOLOv5檢測代碼...
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # 目錄設置
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 加載模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 數據加載
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推理準備
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 處理預測結果
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
            current_objs = list(set(names[int(cls)] for *_, _, cls in det)) # 獲取當前偵測到的物體類別列表

            # 只在 Gemini 啟用、達到間隔且偵測到物體列表有變化時才呼叫 Gemini
            # 或者即使沒變化，但超過一定時間也重新查詢？這裡沿用只在變化時查詢的邏輯
            if gemini_model and (time.time() - last_gemini_time) >= gemini_interval and current_objs != last_detected:
                 if current_objs: # 確保 current_objs 不為空，避免無物體時觸發 Gemini
                    try:
                        # === 定義並呼叫 Gemini API ===
                        # 使用之前討論的最佳化 Prompt
                        prompt = f"""
我正在使用 YOLOv5 偵測物體。根據偵測到的以下物體列表：{', '.join(current_objs)}，我需要一個給樹莓派自動化系統的指令訊息。

請只回傳一個 JSON 格式的字串，不要包含任何額外的文字或解釋。

這個 JSON 物件必須包含以下鍵：
"event": 表示事件類型 (例如: "detection", "status")
"objects": 包含偵測到的物體名稱列表 (例如: ["person", "cat"])
"action": "action": 將action設定成跟objects一樣的名字 ，全部字母小寫，若是有thumbs的存在把s去掉

例如，如果偵測到一個人，理想的回應是 `{{\"event\": \"detection\", \"objects\": [\"person\"], \"action\": \"person\"}}`
如果偵測到一隻貓和一隻狗，理想的回應是 `{{\"event\": \"detection\", \"objects\": [\"cat\", \"dog\"], \"action\": \"dog\"}}`
如果偵測到一張椅子，理想的回應是 `{{\"event\": \"detection\", \"objects\": [\"chair\"], \"action\": \"chair\"}}`
如果沒有偵測到物體，或者偵測到的物體不重要，可以建議動作為 "none"。

請嚴格只回傳 JSON 字串。
"""
                        print(f"\n[Gemini Prompt]: {prompt[:200]}...") # 印出 Prompt 的一部分方便偵錯
                        response = gemini_model.generate_content(prompt)
                        # === Gemini API 呼叫結束 ===

                        # === 處理 Gemini 回應 ===
                        # 檢查回應是否有效且非安全攔截
                        if response.candidates and response.candidates[0].finish_reason == 1:
                            raw_gemini_text = response.candidates[0].content.parts[0].text
                            print(f"\n[Gemini Returned Raw Text]: {raw_gemini_text}") # 印出原始回傳文本

                            # === 提取純粹的 JSON 字串並解析 ===
                            extracted_json_string = ""
                            try:
                                # 找到第一個 { 和最後一個 } 的索引
                                json_start_index = raw_gemini_text.find('{')
                                json_end_index = raw_gemini_text.rfind('}')

                                if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                                    # 提取從第一個 { 到最後一個 } 的子字串
                                    extracted_json_string = raw_gemini_text[json_start_index : json_end_index + 1]

                                    # 解析提取出的純 JSON 字串為 Python 字典
                                    gemini_data = json.loads(extracted_json_string)
                                    print(f"\n[Gemini Parsed JSON]: {gemini_data}")

                                    # === 發送 HTTP POST 請求到樹莓派 ===
                                    try:
                                        # 將當前時間戳加入到要發送的數據中
                                        current_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        gemini_data["timestamp"] = current_timestamp_str # 新增或覆蓋 timestamp

                                        # 將字典數據作為 JSON 格式發送
                                        response_pi = requests.post(raspberry_pi_command_url, json=gemini_data, timeout=5) # 設定 timeout

                                        if response_pi.status_code == 200:
                                            LOGGER.info(f"Successfully sent command to Raspberry Pi: {gemini_data}")
                                        else:
                                            LOGGER.warning(f"Failed to send command to Raspberry Pi. Status code: {response_pi.status_code}, Response: {response_pi.text}")

                                    except requests.exceptions.RequestException as e:
                                        LOGGER.warning(f"HTTP request to Raspberry Pi failed: {e}")
                                    # === HTTP 發送結束 ===

                                    # === 更新狀態和顯示文字 (使用解析後的數據) ===
                                    # 在影像上顯示解析後的資訊
                                    display_text = f"Act: {gemini_data.get('action', 'N/A')}, Objs: {', '.join(gemini_data.get('objects', []))}"
                                    cv2.putText(im0, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(im0, f"Time: {gemini_data.get('timestamp', 'N/A')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                                    last_detected = current_objs # 只有成功發送且內容有變化才更新 last_detected
                                    last_gemini_time = time.time() # 只有成功發送才更新時間

                                else:
                                    # 如果沒有找到有效的 JSON 結構
                                    LOGGER.warning(f"Could not find valid JSON structure in Gemini response: {raw_gemini_text}")
                                    # 可以選擇如何處理，例如不更新 last_detected 或重設計時器

                            except json.JSONDecodeError:
                                # 如果提取出的子字串不是有效的 JSON 格式
                                LOGGER.warning(f"Extracted string is not valid JSON: {extracted_json_string}")
                                # 處理 JSON 格式不正確的情況

                            except Exception as e:
                                # 捕獲提取或解析 JSON 時的其他可能錯誤
                                print(f"Error during JSON extraction or parsing: {e}")


                        else:
                            # 如果 Gemini 回應無效或被攔截
                            finish_reason = response.candidates[0].finish_reason if response.candidates else 'N/A'
                            safety_ratings = response.candidates[0].safety_ratings if response.candidates else 'N/A'
                            LOGGER.warning(f"Gemini response was not valid or was blocked. Finish reason: {finish_reason}, Safety ratings: {safety_ratings}")
                            # 如果回應無效，考慮不更新 last_detected，這樣下次達到間隔會重新查詢
                            # last_detected = [] # 或者重設為空列表，強制下次查詢


                    except Exception as e:
                        # 捕獲 Gemini API 呼叫本身的錯誤 (如 429, 連線問題等)
                        print(f"Gemini API call error: {e}")
                        # 如果 API 呼叫失敗，考慮不更新 last_detected 或重設計時器
                        # last_detected = [] # 或者重設為空列表


            # === Gemini互動功能 - 到這裡結束 ===

            # 顯示結果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

# 保存結果
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else: # This is the video/stream saving block (這是影片/串流儲存區塊)
                    # 加入這個檢查，確保索引 i 不會超出 vid_path/vid_writer 的範圍
                    if i < bs: # <-- 在這裡加入判斷
                        # print(f"Current index i: {i}") # 您的偵錯列印可以保留或移除
                        if vid_path[i] != save_path:
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else: # Default FPS, width, height if vid_cap is None
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            # 使用對應串流索引 i 的 writer
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # 使用對應串流索引 i 的 writer 寫入影格
                        vid_writer[i].write(im0)
                    else:
                        # (可選) 如果發生了意外的索引 i >= bs，可以印出警告
                        LOGGER.warning(f"Unexpected stream index {i} encountered (batch size is {bs}). Skipping video save for this index.")

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])

def parse_opt():
    parser = argparse.ArgumentParser()
    # 原始YOLOv5參數
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    # Gemini專用參數
    parser.add_argument('--gemini-interval', type=int, default=2, help='Gemini響應間隔(秒)')
    parser.add_argument('--disable-gemini', action='store_true', help='禁用Gemini功能')
    parser.add_argument('--gemini-api-key', type=str, default=None, help='Gemini API密鑰')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

import streamlit as st
import os
import json
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from pdf2image import convert_from_path
import cv2
import numpy as np
import mimetypes
import tempfile
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND") or st.secrets.get('GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND')
if os.path.isfile(".env") or os.getenv("ENV") == "dev":  
    if GOOGLE_CREDENTIALS and os.path.isfile(GOOGLE_CREDENTIALS):
        with open(GOOGLE_CREDENTIALS, "r") as f:
            DEFAULT_SA_KEY = json.load(f)
    else:
        st.error("本地環境：請在 .env 中設定 GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND 為有效 JSON 檔案路徑")
        DEFAULT_SA_KEY = None
else:  # 正式環境（如 Streamlit Cloud）
    if GOOGLE_CREDENTIALS:
        if isinstance(GOOGLE_CREDENTIALS, dict):
            DEFAULT_SA_KEY = GOOGLE_CREDENTIALS  
        else:
            try:
                DEFAULT_SA_KEY = json.loads(GOOGLE_CREDENTIALS)  
            except (json.JSONDecodeError, TypeError):
                st.error("正式環境：GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND 必須是有效的 JSON 字串或字典")
                DEFAULT_SA_KEY = None
    else:
        DEFAULT_SA_KEY = None

# 設置 GOOGLE_APPLICATION_CREDENTIALS
if DEFAULT_SA_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(DEFAULT_SA_KEY, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
else:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error("未找到有效的 Google Cloud 憑證，請檢查環境設定")
        st.stop()

st.title("Document AI: Signature Field Detection")
st.markdown("**Kdan Bennett**")
st.markdown("Extract the signature field from your doc")

# 側邊欄上傳憑證（覆蓋預設）
st.sidebar.subheader("Upload GCP JSON Key (Optional)")
st.sidebar.write("Default credentials are in use. Upload your own JSON key to override.")
sa_key = st.sidebar.file_uploader("Upload JSON 文件", type=["json"], key="sa_key")

if sa_key:
    custom_key = json.load(sa_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(custom_key, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
    st.sidebar.success("JSON KEY HAS BEEN UPLOADED (Overriding default)")

try:
    documentai_client = documentai.DocumentProcessorServiceClient()
    storage_client = storage.Client()
except Exception as e:
    st.error(f"無法初始化 GCP 客戶端，請檢查憑證：{e}")
    st.stop()

processor_name = "projects/962438265955/locations/us/processors/6d0867440d8644c3"
BUCKET_NAME = "dataset_signature"

# 初始化 session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'boxes' not in st.session_state:
    st.session_state.boxes = None
if 'gcs_path' not in st.session_state:
    st.session_state.gcs_path = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = "doc_upload"

st.subheader("Upload Target File")
st.write("Only support format with PDF (within 15 pages), JPG, JPEG, PNG")
st.info("上傳的文件將用於簽名檢測並儲存以優化服務體驗。請避免上傳包含敏感或機密資訊的文件。")
uploaded_file = st.file_uploader("SELECT FILE", type=["pdf", "jpg", "jpeg", "png"], key=st.session_state.uploader_key)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    st.write(f"已選擇文件：{st.session_state.uploaded_file.name}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("開始"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file.read())
                file_path = tmp_file.name

            # 在正式環境中上傳文件到 GCS
            if not (os.path.isfile(".env") or os.getenv("ENV") == "dev"):
                try:
                    bucket = storage_client.bucket(BUCKET_NAME)
                    now = datetime.now()
                    upload_date = now.strftime("%Y%m%d")
                    timestamp = now.strftime("%H%M%S")
                    mime_type, _ = mimetypes.guess_type(st.session_state.uploaded_file.name)
                    file_type = "pdf" if mime_type == "application/pdf" else "image"
                    original_filename = st.session_state.uploaded_file.name
                    gcs_blob_name = f"streamlit_dataset/{upload_date}/{file_type}/{timestamp}_{original_filename}"
                    blob = bucket.blob(gcs_blob_name)
                    blob.upload_from_filename(file_path)
                    st.session_state.gcs_path = f"gs://{BUCKET_NAME}/{gcs_blob_name}"
                    st.success(f"文件已成功上傳至 GCS: {st.session_state.gcs_path}")
                except Exception as e:
                    st.error(f"無法上傳文件至 GCS: {e}")

            def detect_signature_boxes(file_path):
                try:
                    mime_type, _ = mimetypes.guess_type(file_path)
                    with open(file_path, "rb") as file:
                        content = file.read()
                    
                    if mime_type == "application/pdf":
                        request = documentai.ProcessRequest(
                            name=processor_name,
                            raw_document=documentai.RawDocument(content=content, mime_type="application/pdf")
                        )
                    elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                        request = documentai.ProcessRequest(
                            name=processor_name,
                            raw_document=documentai.RawDocument(content=content, mime_type=mime_type)
                        )
                    else:
                        st.error("FORMAT INVALID")
                        return []

                    result = documentai_client.process_document(request=request)
                    boxes = []
                    for entity in result.document.entities:
                        if entity.type_ == "signature_field":
                            for page_ref in entity.page_anchor.page_refs:
                                bounding_box = page_ref.bounding_poly.normalized_vertices
                                box = [bounding_box[0].x, bounding_box[0].y, bounding_box[2].x, bounding_box[2].y]
                                boxes.append({"box": box, "confidence": entity.confidence, "page": page_ref.page})
                    return boxes
                except Exception as e:
                    st.error(f"detect_signature_boxes ERROR: {e}")
                    return []

            def find_nearest_underline(image, text_box, search_range=50):
                """
                尋找離動態文本最近的底線，並返回其上方區域作為簽名區域
                """
                x_min, y_min, x_max, y_max = map(int, text_box)
                
                # 定義搜索區域（右邊和下方）
                areas = {
                    "right": (x_max, y_min - search_range, 1327, y_max + search_range),
                    "bottom": (x_min, y_max, x_max, y_max + search_range)
                }
                
                closest_line = None
                min_distance = float('inf')
                best_direction = None
                all_lines = []
                
                for direction, (a_x_min, a_y_min, a_x_max, a_y_max) in areas.items():
                    if a_x_max <= a_x_min or a_y_max <= a_y_min:
                        continue
                    search_area = image[max(0, a_y_min):a_y_max, max(0, a_x_min):a_x_max]
                    if search_area.size == 0:
                        continue
                    
                    # 轉為灰度圖並二值化
                    gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                    
                    # 尋找水平線（底線），調整參數以檢測更長的底線
                    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=10, minLineLength=200, maxLineGap=150)
                    if lines is None:
                        continue
                    
                    # 收集所有線段
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(y1 - y2) < 10:  # 確保是水平線
                            if direction == "right":
                                distance = abs(y1 - (y_max - a_y_min))
                            else:  # bottom
                                distance = abs(y1)
                            all_lines.append((x1, y1, x2, y2, distance, direction))
                
                if not all_lines:
                    return text_box
                
                # 按距離排序，選擇最近的底線
                all_lines.sort(key=lambda x: x[4])
                closest_line = all_lines[0]
                min_distance = closest_line[4]
                best_direction = closest_line[5]
                
                # 合併同一高度的線段
                merged_lines = []
                current_line = list(closest_line[:4])
                current_y = closest_line[1]
                for line in all_lines[1:]:
                    x1, y1, x2, y2, _, direction = line
                    if direction != best_direction:
                        continue
                    if abs(y1 - current_y) < 10:  # 同一高度
                        current_line[0] = min(current_line[0], x1)
                        current_line[2] = max(current_line[2], x2)
                    else:
                        merged_lines.append(current_line)
                        current_line = [x1, y1, x2, y2]
                        current_y = y1
                merged_lines.append(current_line)
                
                # 選擇最長的底線
                longest_line = max(merged_lines, key=lambda line: line[2] - line[0])
                line_x1, line_y1, line_x2, line_y2 = longest_line
                
                # 轉換回原圖座標
                if best_direction == "right":
                    line_y = line_y1 + (y_min - search_range)
                    line_x_min = line_x1 + x_max
                    line_x_max = line_x2 + x_max
                else:  # bottom
                    line_y = line_y1 + y_max
                    line_x_min = line_x1 + x_min
                    line_x_max = line_x2 + x_min
                
                # 從底線往上檢測空白區域，確定簽名區域高度
                signature_height = 0
                max_height = 50
                for h in range(1, max_height + 1):
                    y_top = line_y - h
                    if y_top < 0:
                        break
                    roi = image[y_top:line_y, line_x_min:line_x_max]
                    if roi.size == 0:
                        break
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    white_pixels = cv2.countNonZero(binary)
                    total_pixels = roi.shape[0] * roi.shape[1]
                    white_ratio = white_pixels / total_pixels
                    if white_ratio < 0.95:
                        break
                    signature_height = h
                
                signature_height = max(30, signature_height)
                signature_box = [line_x_min, line_y - signature_height, line_x_max, line_y]
                return signature_box


            def find_signature_area(image, text_box, search_range=50):
                """
                動態判斷簽名區域位置（直接尋找底線）
                """
                x_min, y_min, x_max, y_max = map(int, text_box)
                return find_nearest_underline(image, text_box, search_range)


            def extract_fillable_area(image, box):
                """
                用二值化提取填寫區域（確保與底線長度一致）
                """
                x_min, y_min, x_max, y_max = map(int, box)
                roi = image[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    return box
                
                # 轉為灰度圖並二值化
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # 尋找輪廓，只保留空白區域
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return box
                
                # 選擇最大的輪廓作為填寫區域
                max_area = 0
                best_contour = None
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    if area > max_area and w > 20 and h > 10:
                        max_area = area
                        best_contour = (x, y, w, h)
                
                if best_contour:
                    x, y, w, h = best_contour
                    fillable_box = [x_min, y_min + y, x_max, y_min + y + h]
                    return fillable_box
                
                return box


            def visualize_boxes(file_path, boxes):
                mime_type, _ = mimetypes.guess_type(file_path)

                if mime_type == "application/pdf":
                    images = convert_from_path(file_path, dpi=200)
                elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                    images = [cv2.imread(file_path)]
                else:
                    st.error("FORMAT DOESN'T SUPPORT VISUALIZATION!")
                    return

                for i, img in enumerate(images):
                    if mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                        img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    page_num = i
                    page_boxes = [box for box in boxes if box["page"] == page_num]
                    height, width = img_cv.shape[:2]

                    for box_info in page_boxes:
                        box = box_info["box"]
                        confidence = box_info["confidence"]
                        x_min, y_min, x_max, y_max = [int(coord * dim) for coord, dim in zip(box, [width, height, width, height])]
                        
                        # 空間分析：動態判斷簽名區域位置
                        signature_box = find_signature_area(img_cv, [x_min, y_min, x_max, y_max], search_range=50)
                        
                        # 二值化：提取填寫區域
                        fillable_box = extract_fillable_area(img_cv, signature_box)
                        x_min_f, y_min_f, x_max_f, y_max_f = fillable_box
                        
                        # 可視化原始框（動態文本）
                        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 紅色框
                        # 可視化填寫區域
                        cv2.rectangle(img_cv, (x_min_f, y_min_f), (x_max_f, y_max_f), (0, 255, 0), 2)  # 綠色框
                        label = f"Conf: {confidence:.2f}"
                        cv2.putText(img_cv, label, (x_min_f, y_min_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption=f"第 {page_num + 1} 頁", use_container_width=True)

            with st.spinner("ANALYSING..."):
                boxes = detect_signature_boxes(file_path)
                st.session_state.boxes = boxes
                if boxes:
                    st.success("SUCCESSFULLY DETECTED SIGNATURE FIELDS!")
                    st.write("Bounding Boxes: ", boxes)
                else:
                    st.warning("沒找到簽名欄位，或者處理過程中出了點問題。")
                
                st.subheader("Visualization")
                visualize_boxes(file_path, boxes)
    
    with col2:
        if st.button("Clear"):
            st.session_state.uploaded_file = None
            st.session_state.boxes = None
            st.session_state.gcs_path = None
            st.session_state.uploader_key = f"doc_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.rerun()
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
from typing import List, Dict, Tuple, Optional

load_dotenv()

# --- Google Cloud Credentials Setup ---
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND") or st.secrets.get('GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND')
if os.path.isfile(".env") or os.getenv("ENV") == "dev":  
    if GOOGLE_CREDENTIALS and os.path.isfile(GOOGLE_CREDENTIALS):
        with open(GOOGLE_CREDENTIALS, "r") as f:
            DEFAULT_SA_KEY = json.load(f)
    else:
        st.error("Local environment: Please set GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND to a valid JSON file path in .env")
        DEFAULT_SA_KEY = None
else:  # Production environment (e.g., Streamlit Cloud)
    if GOOGLE_CREDENTIALS:
        if isinstance(GOOGLE_CREDENTIALS, dict):
            DEFAULT_SA_KEY = GOOGLE_CREDENTIALS  
        else:
            try:
                DEFAULT_SA_KEY = json.loads(GOOGLE_CREDENTIALS)  
            except (json.JSONDecodeError, TypeError):
                st.error("Production environment: GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND must be a valid JSON string or dict")
                DEFAULT_SA_KEY = None
    else:
        DEFAULT_SA_KEY = None

if DEFAULT_SA_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(DEFAULT_SA_KEY, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
else:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error("No valid Google Cloud credentials found. Please check environment settings.")
        st.stop()

# --- Streamlit UI Setup ---
st.title("Document AI: Signature Field Detection")
st.markdown("**Kdan Bennett**")
st.markdown("""
### IDP Signature Field Detection (trial version)
**Summary**：  
此系統由 Kdan Infra 開發，通過自動檢測 PDF 和圖像（JPG、PNG）中的簽名欄位來簡化文件處理流程。
系統利用 Google Cloud 的 Document AI 提取結構化數據並推斷簽名區域，提升文件處理效率。

**解決的痛點**：  
- **手動審查耗時**：傳統方法需人工定位簽名欄位，拖慢工作流程。  
- **檢測不一致**：文件佈局多樣化導致簽名欄位識別難以標準化。  
- **擴展性問題**：手動處理難以應對大量文件。

**Process**：  
1. **上傳**：用戶上傳 PDF 或圖像文件。  
2. **OCR 與實體提取**：Document AI 處理文件，識別簽名欄位實體。  
3. **簽名區域推斷**：根據標籤文字與周圍標記，自定義邏輯推斷簽名位置。  
4. **視覺化**：顯示結果，標籤框為紅色，簽名區域為綠色。

**Future**：  
這是MVP版本。後續將引入佈局感知的多模態模型（結合文字、圖像與空間數據），提升準確性並處理複雜文件結構。這部分會需要花更多時間及計算資源來完成，未來會訓練本地模型來取代雲端模型。

**Pricing**：  

 USD$ 30 / 1000 pages = USD$ 0.03/page

**注意**：目前支援 PDF（最多 15 頁）和圖像文件。上傳文件將用於服務優化，請避免包含敏感資訊。
""")

# Sidebar for optional GCP JSON key upload
st.sidebar.subheader("Upload GCP JSON Key (Optional)")
st.sidebar.text("Default credentials are in use. Upload your own JSON key to override.")
sa_key = st.sidebar.file_uploader("Upload JSON 文件", type=["json"], key="sa_key")

if sa_key:
    custom_key = json.load(sa_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(custom_key, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
    st.sidebar.success("JSON KEY HAS BEEN UPLOADED (Overriding default)")

# --- Initialize Google Cloud Clients ---
try:
    documentai_client = documentai.DocumentProcessorServiceClient()
    storage_client = storage.Client()
except Exception as e:
    st.error(f"Failed to initialize GCP clients. Please check credentials: {e}")
    st.stop()

processor_name = "projects/962438265955/locations/us/processors/6d0867440d8644c3"
BUCKET_NAME = "dataset_signature"

# --- Session State Initialization ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'boxes' not in st.session_state:
    st.session_state.boxes = None
if 'gcs_path' not in st.session_state:
    st.session_state.gcs_path = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = "doc_upload"

# --- File Upload UI ---
st.subheader("Upload Target File")
st.text("Only support format with PDF (within 15 pages), JPG, JPEG, PNG")
st.info("上傳的文件將用於簽名檢測並儲存以優化服務體驗。請避免上傳包含敏感或機密資訊的文件。")
uploaded_file = st.file_uploader("SELECT FILE", type=["pdf", "jpg", "jpeg", "png"], key=st.session_state.uploader_key)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# --- Core Functions from Your Code (Unchanged) ---

def get_pixel_bbox(normalized_vertices: List[documentai.NormalizedVertex], page_width: int, page_height: int) -> Optional[Tuple[int, int, int, int]]:
    """將 Document AI 的 Normalized Vertices 轉換為像素 BBox [xmin, ymin, xmax, ymax]。"""
    if not normalized_vertices:
        return None
    try:
        x_coords = [v.x for v in normalized_vertices if v.x is not None and 0 <= v.x <= 1]
        y_coords = [v.y for v in normalized_vertices if v.y is not None and 0 <= v.y <= 1]
        if not x_coords or not y_coords:
            return None
        xmin = np.floor(min(x_coords) * page_width)
        ymin = np.floor(min(y_coords) * page_height)
        xmax = np.ceil(max(x_coords) * page_width)
        ymax = np.ceil(max(y_coords) * page_height)
        if any(coord is None or not np.isfinite(coord) for coord in [xmin, ymin, xmax, ymax]):
            return None
        if xmax <= xmin or ymax <= ymin:
            if xmax < xmin or ymax < ymin:
                return None
        return int(xmin), int(ymin), int(xmax), int(ymax)
    except Exception as e:
        return None

def get_page_tokens(page: documentai.Document.Page, page_width: int, page_height: int) -> List[Dict]:
    """提取頁面中所有 token 的文本和像素 BBox。"""
    tokens = []
    if not page.tokens:
        return tokens
    for token in page.tokens:
        if not token.layout or not token.layout.text_anchor or not token.layout.text_anchor.content: continue
        text = token.layout.text_anchor.content
        if not token.layout.bounding_poly or not token.layout.bounding_poly.normalized_vertices: continue
        pixel_bbox = get_pixel_bbox(token.layout.bounding_poly.normalized_vertices, page_width, page_height)
        if pixel_bbox:
            if pixel_bbox[2] > pixel_bbox[0] and pixel_bbox[3] > pixel_bbox[1]:
                tokens.append({
                    "text": text,
                    "bbox": pixel_bbox,
                    "raw_token": token
                })
    return tokens

def find_nearest_token_on_line(
    target_bbox: Tuple[int, int, int, int],
    page_tokens: List[Dict],
    direction: str = 'right',
    y_tolerance_factor: float = 0.7,
    max_horizontal_dist: int = 800,
    ignore_chars: str = ":： ",
    debug_mode: bool = False
) -> Optional[Dict]:
    tx_min, ty_min, tx_max, ty_max = target_bbox
    target_cy = (ty_min + ty_max) / 2
    target_height = ty_max - ty_min
    if target_height <= 0: target_height = 1
    nearest_token_info = None
    min_dist = float('inf')
    for token_info in page_tokens:
        tok_bbox = token_info["bbox"]
        tok_x_min, tok_y_min, tok_x_max, tok_y_max = tok_bbox
        tok_cy = (tok_y_min + tok_y_max) / 2
        tok_text = token_info["text"]
        tok_text_stripped = token_info["text"].strip(ignore_chars)
        if not tok_text_stripped:
            continue
        y_distance = abs(tok_cy - target_cy)
        is_on_line = y_distance < (target_height * y_tolerance_factor)
        if is_on_line:
            dist = float('inf')
            valid_direction = False
            gap_threshold = -2
            if direction == 'right' and tok_x_min >= tx_max + gap_threshold:
                dist = tok_x_min - tx_max
                valid_direction = True
            elif direction == 'left' and tok_x_max <= tx_min - gap_threshold:
                dist = tx_min - tok_x_max
                valid_direction = True
            current_distance_metric = max(0, dist)
            if valid_direction and current_distance_metric < max_horizontal_dist and current_distance_metric < min_dist:
                min_dist = current_distance_metric
                nearest_token_info = token_info
    return nearest_token_info

BELOW_LABEL_KEYWORDS = ["簽章", "簽名", "蓋章", "Seal", "Signature"]
DEFAULT_SIGNATURE_AREA_HEIGHT_FACTOR_BELOW = 1.8
DEFAULT_MIN_WIDTH_FACTOR_BELOW = 3.0
DEFAULT_BELOW_VERTICAL_MARGIN = 5
_HORIZONTAL_MARGIN = 5
_MIN_INFERRED_HEIGHT = 15

def infer_signature_area_bbox(
    label_entity: documentai.Document.Entity,
    page_tokens: List[Dict],
    page_width: int,
    page_height: int,
    min_sig_width: int = 50,
    signature_area_height_factor: float = 0.8,
    enable_debug_prints: bool = True,
    default_width_factor_of_height: float = 7.0,
    max_absolute_default_width: int = 450,
    max_relative_default_width_factor: float = 0.6,
    nearest_token_max_dist: int = 1000,
    nearest_token_y_factor: float = 0.75,
    signature_area_height_factor_below: float = DEFAULT_SIGNATURE_AREA_HEIGHT_FACTOR_BELOW,
    min_width_factor_below: float = DEFAULT_MIN_WIDTH_FACTOR_BELOW,
    vertical_margin_below: int = DEFAULT_BELOW_VERTICAL_MARGIN,
) -> Optional[Tuple[Tuple[int, int, int, int], str]]:
    if not label_entity.page_anchor or not label_entity.page_anchor.page_refs:
        return None
    page_ref = label_entity.page_anchor.page_refs[0]
    if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
        return None
    label_bbox = get_pixel_bbox(page_ref.bounding_poly.normalized_vertices, page_width, page_height)
    if label_bbox is None:
        return None
    lx_min, ly_min, lx_max, ly_max = label_bbox
    label_text = label_entity.mention_text if label_entity.mention_text else ""
    line_height = ly_max - ly_min
    if line_height <= 0: line_height = 20
    sig_xmin, sig_ymin, sig_xmax, sig_ymax = -1, -1, -1, -1
    scene = "unknown"
    horizontal_margin = _HORIZONTAL_MARGIN

    def calculate_dynamic_default_width(start_x: int, direction: str) -> int:
        if direction == 'right':
            available_space = page_width - start_x - horizontal_margin
        else:
            available_space = start_x - horizontal_margin
        dynamic_w = line_height * default_width_factor_of_height
        capped_w = min(dynamic_w, max_absolute_default_width, max(0, available_space) * max_relative_default_width_factor)
        final_w = max(capped_w, min_sig_width)
        return int(final_w)

    is_below_case = False
    label_stripped = label_text.strip()
    label_ends_with_keyword = any(label_stripped.endswith(kw) for kw in BELOW_LABEL_KEYWORDS)
    if label_ends_with_keyword:
        short_check_dist = max(50, int(line_height * 1.5))
        nearest_right_short_dist_check = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=short_check_dist,
            y_tolerance_factor=0.6,
            ignore_chars=":： ",
            debug_mode=False
        )
        if nearest_right_short_dist_check is None:
            is_below_case = True
            scene = "below_label"

    if is_below_case:
        sig_ymin = ly_max + vertical_margin_below
        estimated_height = int(line_height * signature_area_height_factor_below)
        estimated_height = max(estimated_height, _MIN_INFERRED_HEIGHT * 1.5)
        sig_ymax = sig_ymin + estimated_height
        sig_xmin = lx_min
        sig_xmax = lx_max
        min_req_width = max(min_sig_width, int(line_height * min_width_factor_below))
        current_width = sig_xmax - sig_xmin
        if current_width < min_req_width:
            needed_expansion = min_req_width - current_width
            expand_left = needed_expansion // 2
            potential_xmin = sig_xmin - expand_left
            actual_xmin = max(0, potential_xmin)
            actual_left_expansion = sig_xmin - actual_xmin
            expand_right = needed_expansion - actual_left_expansion
            potential_xmax = sig_xmax + expand_right
            actual_xmax = min(page_width, potential_xmax)
            sig_xmin = actual_xmin
            sig_xmax = actual_xmax
            final_width = sig_xmax - sig_xmin
            if final_width < min_sig_width * 0.9:
                pass
            scene += "_expand_width"

    elif label_stripped.endswith(':') or label_stripped.endswith('：'):
        scene = "right_colon"
        sig_xmin = lx_max + horizontal_margin
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        nearest_right_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        if nearest_right_token:
            nr_bbox = nearest_right_token["bbox"]
            sig_xmax = max(sig_xmin + min_sig_width // 2, nr_bbox[0] - horizontal_margin)
        else:
            dynamic_width = calculate_dynamic_default_width(sig_xmin, 'right')
            sig_xmax = min(sig_xmin + dynamic_width, page_width - horizontal_margin)

    elif label_stripped.startswith('(') and label_stripped.endswith(')'):
        scene = "left_paren"
        sig_xmax = lx_min - horizontal_margin
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        nearest_left_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='left',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        if nearest_left_token:
            nl_bbox = nearest_left_token["bbox"]
            sig_xmin = min(sig_xmax - min_sig_width // 2, nl_bbox[2] + horizontal_margin)
        else:
            dynamic_width = calculate_dynamic_default_width(sig_xmax, 'left')
            sig_xmin = max(sig_xmax - dynamic_width, horizontal_margin)

    else:
        scene = "fallback_right"
        sig_xmin = lx_max + horizontal_margin
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        nearest_right_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        if nearest_right_token:
            nr_bbox = nearest_right_token["bbox"]
            sig_xmax = max(sig_xmin + min_sig_width // 2, nr_bbox[0] - horizontal_margin)
        else:
            dynamic_width = calculate_dynamic_default_width(sig_xmin, 'right')
            sig_xmax = min(sig_xmin + dynamic_width, page_width - horizontal_margin)

    if sig_xmin == -1 or sig_xmax == -1 or sig_ymin == -1 or sig_ymax == -1:
        return None
    calculated_width = sig_xmax - sig_xmin
    if calculated_width < min_sig_width:
        if scene.startswith("left"):
            sig_xmin = max(0, sig_xmax - min_sig_width)
        else:
            sig_xmax = min(page_width, sig_xmin + min_sig_width)
        if sig_xmax - sig_xmin < min_sig_width * 0.8:
            pass
        else:
            scene += "_final_min_width_fix"
    sig_xmin = max(0, sig_xmin)
    sig_ymin = max(0, sig_ymin)
    sig_xmax = min(page_width, sig_xmax)
    sig_ymax = min(page_height, sig_ymax)
    if sig_ymax <= sig_ymin:
        sig_ymin = ly_max + 2
        sig_ymax = sig_ymin + max(int(line_height * 1.5), _MIN_INFERRED_HEIGHT)
        sig_ymin = max(0, sig_ymin)
        sig_ymax = min(page_height, sig_ymax)
        if sig_ymax <= sig_ymin:
            return None
        scene += "_y_final_fallback"
    inferred_bbox = (int(sig_xmin), int(sig_ymin), int(sig_xmax), int(sig_ymax))
    if inferred_bbox[2] <= inferred_bbox[0] or inferred_bbox[3] <= inferred_bbox[1]:
        return None
    return inferred_bbox, scene

def process_document_ai(file_path: str) -> Optional[documentai.Document]:
    global documentai_client, processor_name
    if not os.path.exists(file_path):
        st.error(f"錯誤: 文件不存在 {file_path}")
        return None
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext == ".pdf": mime_type = "application/pdf"
            elif ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            elif ext == ".png": mime_type = "image/png"
            elif ext == ".tiff" or ext == ".tif": mime_type = "image/tiff"
            else: raise ValueError(f"無法確定文件類型: {file_path}")
        supported_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg", "image/tiff"]
        if mime_type not in supported_types:
            raise ValueError(f"不支持的文件類型: {mime_type}。請提供 PDF, JPG, PNG, TIFF 文件。")
        with open(file_path, "rb") as file:
            content = file.read()
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )
        result = documentai_client.process_document(request=request)
        return result.document
    except Exception as e:
        st.error(f"處理文件 {file_path} 時發生錯誤: {e}")
        return None

def extract_and_infer_signature_areas(document: documentai.Document) -> Tuple[List[Dict], Dict]:
    results = []
    doc_ai_dimensions = {}
    if not document.pages:
        return results, doc_ai_dimensions
    page_tokens_map = {}
    for i, page in enumerate(document.pages):
        if not page.dimension or not page.dimension.width or not page.dimension.height or page.dimension.width <= 0 or page.dimension.height <= 0:
            continue
        page_width_docai = int(page.dimension.width)
        page_height_docai = int(page.dimension.height)
        doc_ai_dimensions[i] = {'width': page_width_docai, 'height': page_height_docai}
        page_tokens_map[i] = {
            "tokens": get_page_tokens(page, page_width_docai, page_height_docai),
            "width": page_width_docai,
            "height": page_height_docai
        }
    if not document.entities:
        return results, doc_ai_dimensions
    target_entity_type = "signature_field"
    processed_entity_count = 0
    min_sig_width_default = 50
    for index, entity in enumerate(document.entities):
        if entity.type_ == target_entity_type:
            if not entity.page_anchor or not entity.page_anchor.page_refs:
                continue
            page_ref = entity.page_anchor.page_refs[0]
            page_num = page_ref.page
            if page_num not in page_tokens_map:
                continue
            page_info = page_tokens_map[page_num]
            page_width_docai = page_info["width"]
            page_height_docai = page_info["height"]
            page_tokens = page_info["tokens"]
            if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
                continue
            label_bbox_normalized = page_ref.bounding_poly.normalized_vertices
            label_bbox_pixel_docai = get_pixel_bbox(label_bbox_normalized, page_width_docai, page_height_docai)
            if label_bbox_pixel_docai is None:
                continue
            inference_result = infer_signature_area_bbox(
                label_entity=entity,
                page_tokens=page_tokens,
                page_width=page_width_docai,
                page_height=page_height_docai,
                min_sig_width=min_sig_width_default,
                enable_debug_prints=False
            )
            inferred_area_bbox_pixel_docai = None
            inferred_rule = "failed"
            if inference_result:
                inferred_area_bbox_pixel_docai, inferred_rule = inference_result
                processed_entity_count += 1
            results.append({
                "index": index,
                "predict_boundingbox": label_bbox_pixel_docai,
                "label_text": entity.mention_text,
                "signature_area_boundingbox": inferred_area_bbox_pixel_docai,
                "page": page_num
            })
    return results, doc_ai_dimensions

# --- Main Processing Logic ---
if st.session_state.uploaded_file:
    st.text(f"已選擇文件：{st.session_state.uploaded_file.name}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("開始"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file.read())
                file_path = tmp_file.name
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
                    st.success(f"File uploaded to GCS: {st.session_state.gcs_path}")
                except Exception as e:
                    st.error(f"Failed to upload file to GCS: {e}")

            # --- Adjusted Visualization for Streamlit ---
            def visualize_results(file_path: str, processed_results: List[Dict], doc_ai_dimensions: Dict):
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type == "application/pdf":
                    images_pil = convert_from_path(file_path, dpi=200)
                    images_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images_pil]
                elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                    img_cv = cv2.imread(file_path)
                    images_cv = [img_cv]
                else:
                    st.error("FORMAT DOESN'T SUPPORT VISUALIZATION!")
                    return
                for page_num, img_cv in enumerate(images_cv):
                    if img_cv is None or img_cv.size == 0:
                        continue
                    if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
                        if len(img_cv.shape) == 2:
                            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                        elif img_cv.shape[2] == 4:
                            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
                        else:
                            img_cv = img_cv[:, :, :3]
                    vis_height, vis_width, _ = img_cv.shape
                    doc_page_width = doc_ai_dimensions.get(page_num, {}).get('width', vis_width)
                    doc_page_height = doc_ai_dimensions.get(page_num, {}).get('height', vis_height)
                    scale_x = vis_width / doc_page_width if doc_page_width > 0 else 1.0
                    scale_y = vis_height / doc_page_height if doc_page_height > 0 else 1.0
                    page_results = [res for res in processed_results if res.get("page") == page_num]
                    if not page_results:
                        continue
                    img_to_show = img_cv.copy()
                    for result in page_results:
                        lx_vis, ly_vis, lx2_vis, ly2_vis = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate(result["predict_boundingbox"])]
                        if lx_vis < lx2_vis and ly_vis < ly2_vis:
                            cv2.rectangle(img_to_show, (lx_vis, ly_vis), (lx2_vis, ly2_vis), (0, 0, 255), 2)
                            text_y = ly_vis - 10 if ly_vis > 15 else ly2_vis + 20
                            cv2.putText(img_to_show, f"Idx: {result['index']}", (lx_vis, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if result["signature_area_boundingbox"]:
                            ix_vis, iy_vis, ix2_vis, iy2_vis = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate(result["signature_area_boundingbox"])]
                            if ix2_vis > ix_vis and iy2_vis > iy_vis:
                                cv2.rectangle(img_to_show, (ix_vis, iy_vis), (ix2_vis, iy2_vis), (0, 255, 0), 2)
                                text_y = iy_vis - 10 if iy_vis > 15 else iy2_vis + 20
                                cv2.putText(img_to_show, "SigArea", (ix_vis, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
                    st.image(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB), caption=f"第 {page_num + 1} 頁", use_container_width=True)

            # --- Process and Display ---
            with st.spinner("ANALYSING..."):
                document = process_document_ai(file_path)
                if document:
                    processed_results, doc_ai_dimensions = extract_and_infer_signature_areas(document)
                    boxes = processed_results  # Already in the required format
                    st.session_state.boxes = boxes
                    if boxes:
                        st.success("SUCCESSFULLY DETECTED SIGNATURE FIELDS!")
                        st.text("Bounding Boxes:")
                        st.write(boxes)
                    else:
                        st.warning("沒找到簽名欄位，或者處理過程中出了點問題。")
                    st.subheader("Visualization")
                    visualize_results(file_path, boxes, doc_ai_dimensions)
                else:
                    st.error("Failed to process document with Document AI.")

    with col2:
        if st.button("Clear"):
            st.session_state.uploaded_file = None
            st.session_state.boxes = None
            st.session_state.gcs_path = None
            st.session_state.uploader_key = f"doc_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.rerun()
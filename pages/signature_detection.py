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
import matplotlib.pyplot as plt
import re

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

 USD $ 30 / 1000 pages = USD $ 0.03/page
 
""")

# Sidebar for optional GCP JSON key upload
st.sidebar.subheader("選擇文件來源")
file_source = st.sidebar.selectbox("Select File Source", ["Upload File", "Use Default File"])
st.sidebar.divider()
st.sidebar.subheader("Upload GCP JSON Key (Optional)")
st.sidebar.text("Default credentials are in use. Upload your own JSON key to override.")
sa_key = st.sidebar.file_uploader("Upload JSON 文件", type=["json"], key="sa_key")
st.sidebar.divider()
enable_debug_prints = st.sidebar.checkbox("Enable Debug Mode", value=False)
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

if file_source == "Upload File":
    st.text("Only support format with PDF (within 15 pages), JPG, JPEG, PNG")
    st.text("<------- You can use default PDF on the left sidebar")
    st.info("上傳的文件將用於簽名檢測並儲存以優化服務體驗。請避免上傳包含敏感或機密資訊的文件。")
    uploaded_file = st.file_uploader("SELECT FILE", type=["pdf", "jpg", "jpeg", "png"], key=st.session_state.uploader_key)

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
elif file_source == "Use Default File":
    default_file_path = "pages/references/社會住宅包租代管第4期計畫民眾承租住宅申請書1120621.pdf"
    if os.path.exists(default_file_path):
        st.info(f"使用預設文件: {os.path.basename(default_file_path)}")
        st.session_state.uploaded_file = open(default_file_path, "rb") # Open default file in binary read mode
    else:
        st.error(f"預設文件路徑錯誤: {default_file_path}")
        st.stop()

# --- Core Functions from Your Code (Unchanged) ---
def extract_signature_blank(image: np.ndarray, bbox: Tuple[int, int, int, int], blank_width_factor: float = 1.0, min_blank_width: int = 20) -> Optional[List[Tuple[int, int, int, int]]]:
    try:
        # Crop Signature field
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            st.warning(f"Warning: Invalid Signature field Bounding Box : {bbox}")
            return None

        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        if x2 <= x1 or y2 <= y1:
            st.warning(f"Warning: Invalid Signature field Bounding Box : ({x1}, {y1}, {x2}, {y2})")
            return None

        signature_area = image[y1:y2, x1:x2]

        # 轉為灰度圖
        gray = cv2.cvtColor(signature_area, cv2.COLOR_BGR2GRAY)

        # adaptiveThreshold to get the text area
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # calcalute projection for the 分界線 between text and blank area
        projection = np.sum(binary, axis=0)
        text_end = np.argmax(projection > 0) + 10  # 10 is for buffering 

        # If there is no text at the end , assume blank is satart from middle
        if text_end >= (x2 - x1):
            text_end = int((x2 - x1) / 2)

        # calculate blank area size
        blank_x1 = x1 + text_end
        blank_x2 = x2
        blank_width = blank_x2 - blank_x1
        if blank_width < min_blank_width:
            blank_x1 = x2 - min_blank_width
            blank_x1 = max(x1, blank_x1)  

        # assume the height is same with the signature field
        blank_y1 = y1
        blank_y2 = y2

        # Check
        if blank_x2 <= blank_x1 or blank_y2 <= blank_y1:
            st.warning(f"Warning: Invalid blank Bounding Box :  ({blank_x1}, {blank_y1}, {blank_x2}, {blank_y2})")
            return None

        return [(blank_x1, blank_y1, blank_x2, blank_y2)] # Return signature blank bounding box

    except Exception as e:
        st.error(f"Error : extract_signature_blank: {e}, BBox: {bbox}")
        return None

def get_pixel_bbox(normalized_vertices: List[documentai.NormalizedVertex], page_width: int, page_height: int) -> Optional[Tuple[int, int, int, int]]:
    if not normalized_vertices:
        st.warning(f"Warning : Empty normalized_vertices。")
        return None

    try:
        # Filter out None coordinates and ensure they are within [0, 1]
        x_coords = [v.x for v in normalized_vertices if v.x is not None and 0 <= v.x <= 1]
        y_coords = [v.y for v in normalized_vertices if v.y is not None and 0 <= v.y <= 1]

        if not x_coords or not y_coords:
             st.warning(f"Warning: unable extract the x, y coordinate from vertices: {normalized_vertices}")
             return None

        # Use np.floor for min and np.ceil for max for conservative bounding box
        # min max are the range of the normalized_vertices coordinate
        # multiple page width and height 轉換成pixel
        xmin = np.floor(min(x_coords) * page_width)
        ymin = np.floor(min(y_coords) * page_height)
        xmax = np.ceil(max(x_coords) * page_width)
        ymax = np.ceil(max(y_coords) * page_height)


        # Check for NaN or Inf coordinates after calculation
        if any(coord is None or not np.isfinite(coord) for coord in [xmin, ymin, xmax, ymax]):
             st.error(f"Warning: Invalid coordinates after calculation: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
             return None

        # Check for invalid box dimensions (max <= min)
        if xmax <= xmin or ymax <= ymin:
             # Check if there is negative, allow zero width/height initially, fix later if needed
             if xmax < xmin or ymax < ymin:
                 st.warning(f"Warning: Invalid bounding box after calculation (max < min): xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
                 return None
             # else: # Handle zero width/height cases if necessary, often downstream logic handles this
             #     print(f"Log: BBox is a line or dot: xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
             #     xmax = max(xmax, xmin + 1)
             #     ymax = max(ymax, ymin + 1)
        return int(xmin), int(ymin), int(xmax), int(ymax)

    except Exception as e:
        st.error(f"Error: Calculation get_pixel_bbox : {e}, vertices: {normalized_vertices}")
        return None

def get_page_tokens(page: documentai.Document.Page, page_width: int, page_height: int) -> List[Dict]:
    """Extract tokens and pixel bounding box from all pages."""
    tokens = []
    if not page.tokens:
         st.warning(f"Warning: Page {page.page_number} has not tokens(Empty Page)。") 
         return tokens

    for token in page.tokens:
        # Basic validation, if 少資料then skip
        if not token.layout or not token.layout.text_anchor or not token.layout.text_anchor.content: continue
        text = token.layout.text_anchor.content
        if not token.layout.bounding_poly or not token.layout.bounding_poly.normalized_vertices: continue
        
        # Calculate pixel box 
        pixel_bbox = get_pixel_bbox(token.layout.bounding_poly.normalized_vertices, page_width, page_height)
        if pixel_bbox:
            # Ensure bbox has non-zero width(邊框) and height before adding
            if pixel_bbox[2] > pixel_bbox[0] and pixel_bbox[3] > pixel_bbox[1]:
                tokens.append({
                    "text": text,
                    "bbox": pixel_bbox,
                    "raw_token": token 
                })
    return tokens

def find_nearest_token_on_line(
    target_bbox: Tuple[int, int, int, int],
    page_tokens: List[Dict], # Token list
    direction: str = 'right', 
    y_tolerance_factor: float = 0.7, # determine the token row
    max_horizontal_dist: int = 800, # default pixel 
    ignore_chars: str = ":： ", # Characters to ignore when checking if token is empty
    debug_mode: bool = False
    ) -> Optional[Dict]:
    tx_min, ty_min, tx_max, ty_max = target_bbox
    target_cy = (ty_min + ty_max) / 2 # calculate center point
    target_height = ty_max - ty_min  # calculate height
    if target_height <= 0: target_height = 1 # Avoid division by zero

    nearest_token_info = None # To save the nearest token
    min_dist = float('inf')

    if debug_mode:
        st.info(f"[Debug find_nearest] Target: {target_bbox} CY: {target_cy:.1f}, H: {target_height}, Y-Tol(px): {target_height * y_tolerance_factor:.1f}")
        # logging.info(f"[Debug find_nearest] Target: {target_bbox} CY: {target_cy:.1f}, H: {target_height}, Y-Tol(px): {target_height * y_tolerance_factor:.1f}")

    potential_matches = [] 

    for token_info in page_tokens:
        tok_bbox = token_info["bbox"]
        tok_x_min, tok_y_min, tok_x_max, tok_y_max = tok_bbox
        tok_cy = (tok_y_min + tok_y_max) / 2
        tok_text = token_info["text"]
        tok_text_stripped = token_info["text"].strip(ignore_chars)

        # Skip tokens that are effectively empty after stripping ignore_chars
        if not tok_text_stripped:
            if debug_mode: potential_matches.append({"text": tok_text, "bbox": tok_bbox, "cy": tok_cy, "on_line": False, "reason": "Empty"})
            continue

        # Check vertical alignment: absolute difference in centers < tolerance
        # Find on the same row
        y_distance = abs(tok_cy - target_cy)
        is_on_line = y_distance < (target_height * y_tolerance_factor)

        # For debugging: collect tokens somewhat close vertically
        if debug_mode and y_distance < (target_height * 1.5):
             potential_matches.append({
                 "text": tok_text, "bbox": tok_bbox, "cy": tok_cy,
                 "on_line": is_on_line, "dist_y": y_distance, "reason": ""
                 })
        elif not is_on_line and debug_mode:
             potential_matches.append({"text": tok_text, "bbox": tok_bbox, "cy": tok_cy, "on_line": False, "dist_y": y_distance, "reason": "Off-line"})

        # Calculate distance and direction
        if is_on_line:
            dist = float('inf')
            valid_direction = False
            gap_threshold = -2 # Default(allow overlap)

            # If direction is right, the left of the token is on the target 's right and allow slight overlap  
            if direction == 'right' and tok_x_min >= tx_max + gap_threshold:
                 # Distance is the gap between target's right and token's left
                 dist = tok_x_min - tx_max
                 valid_direction = True
           
            # If direction is left, the right of the token is on the target 's rileftght and allow slight overlap  
            elif direction == 'left' and tok_x_max <= tx_min - gap_threshold:
                 # Distance is the gap between token's right and target's left
                 dist = tx_min - tok_x_max
                 valid_direction = True

            # Ensure distance is non-negative (gap or touching)
            current_distance_metric = max(0, dist)

            # Update the nearest token
            if valid_direction and current_distance_metric < max_horizontal_dist and current_distance_metric < min_dist:
                 min_dist = current_distance_metric
                 nearest_token_info = token_info
                 if debug_mode:
                    # Update reason for debug matches
                    for m in potential_matches:
                        if m["bbox"] == tok_bbox: m["reason"] = f"Potential Match (Dist: {current_distance_metric:.1f})"


    if debug_mode:
        st.info(f"[Debug find_nearest] Tokens near Target Y:")
        # logging.info(f"[Debug find_nearest] Tokens near Target Y:")
        potential_matches.sort(key=lambda x: x['bbox'][0]) # Sort by x-coordinate
        for p_match in potential_matches:
            status = ""
            if p_match.get("on_line"): status += "OnLine "
            if p_match.get("reason"): status += f"({p_match['reason']}) "
            st.info(f"  - Text: '{p_match['text']}', BBox: {p_match['bbox']}, CY: {p_match['cy']:.1f}, DistY: {p_match.get('dist_y', -1):.1f} {status}")
            # logging.info(f"  - Text: '{p_match['text']}', BBox: {p_match['bbox']}, CY: {p_match['cy']:.1f}, DistY: {p_match.get('dist_y', -1):.1f} {status}")
        if nearest_token_info:
            st.info(f"[Debug find_nearest] Selected Nearest: '{nearest_token_info['text']}' BBox: {nearest_token_info['bbox']} DistMetric: {min_dist:.1f}")
            # logging.info(f"[Debug find_nearest] Selected Nearest: '{nearest_token_info['text']}' BBox: {nearest_token_info['bbox']} DistMetric: {min_dist:.1f}")
        else:
            st.info(f"[Debug find_nearest] No suitable nearest token found in direction '{direction}' within {max_horizontal_dist}px.")
            # logging.info(f"[Debug find_nearest] No suitable nearest token found in direction '{direction}' within {max_horizontal_dist}px.")

    return nearest_token_info

def infer_signature_area_bbox(
    label_entity: documentai.Document.Entity, # signature_field label in document ai
    page_tokens: List[Dict], # token list
    page_width: int, # page width
    page_height: int, # page height
    min_sig_width: int = 50,  # minimun signature of the signature area
    signature_area_height_factor: float = 0.8,  # Height for horizontal cases
    enable_debug_prints: bool = False,
    default_width_factor_of_height: float = 7.0,  # Height for horizontal fallback
    max_absolute_default_width: int = 450,  # Max abs width for horizontal fallback
    max_relative_default_width_factor: float = 0.6,  # max relative width for horizontal fallback
    nearest_token_max_dist: int = 1000, # max distance
    nearest_token_y_factor: float = 0.75, # vertical tolerance factor

    # --- Constants for 'right_colon_with_newline_paren' scenario ---
    # etc bounding box = 甲方:__________(簽章) --> to find the blank area between token
    # 換行或空格後跟著括號，例如 "\n(簽章)" 或 " (簽名)"
    NEWLINE_PAREN_PATTERN = re.compile(r"(\n|\s+)\([^()]+?\)"),

    # --- Parameters for 'below' case ---
    BELOW_LABEL_KEYWORDS = ["簽章", "簽名", "蓋章", "Signature"] ,
    signature_area_height_factor_below: float = 1.8, # How many times the label height for the signature area height below
    min_width_factor_below: float = 3.0 ,# Minimum width relative to label height for the area below
    vertical_margin_below: int = 10, # Pixels between label bottom and area top , avoid overlap

    # --- General Scenario
    _HORIZONTAL_MARGIN = 5,
    _MIN_INFERRED_HEIGHT = 15, # Minimum height for signature area

    # --- New parameter for image-based blank detection ---
    image: Optional[np.ndarray] = None,  # 原始圖像，用於二值化提取空白區域
) -> Optional[Tuple[Tuple[int, int, int, int], str]]:

    # --- Get Label BBox ---
    if not label_entity.page_anchor or not label_entity.page_anchor.page_refs:
        if enable_debug_prints: st.warning(f"Warining: Label Entity '{label_entity.mention_text}' no page_refs。")
        # loggin.warning(f"Warining: Label Entity '{label_entity.mention_text}' no page_refs。")
        return None
    # First page contain label coordinate details
    page_ref = label_entity.page_anchor.page_refs[0]

    # Use the bounding poly from page_ref for the label
    if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
         if enable_debug_prints: st.error(f"Error: Label Entity '{label_entity.mention_text}'  page_ref has no bounding_poly。")
        #  logging.error(f"Error: Label Entity '{label_entity.mention_text}'  page_ref has no bounding_poly。")
         return None
    # Get label coordinate bounding box
    label_bbox = get_pixel_bbox(page_ref.bounding_poly.normalized_vertices, page_width, page_height)
    if label_bbox is None:
        if enable_debug_prints: st.error(f"Error: Error extract Label Entity '{label_entity.mention_text}' bounding box。")
        # logging.error(f"Error: Error extract Label Entity '{label_entity.mention_text}' bounding box。")
        return None

    # --- Initialize variables ---
    lx_min, ly_min, lx_max, ly_max = label_bbox
    label_text = label_entity.mention_text if label_entity.mention_text else "" # Handle empty mention_text
    # calculate height
    line_height = ly_max - ly_min
    if line_height <= 0: line_height = 20 # Avoid division by zero or negative height
    if enable_debug_prints: st.info(f"\Handling Label: '{label_text}', Boundingbox: {label_bbox}, H:{line_height}")
    # logging.info(f"\Handling Label: '{label_text}', Boundingbox: {label_bbox}, H:{line_height}")

    # signature area coordinate (隨便設)
    sig_xmin, sig_ymin, sig_xmax, sig_ymax = -1, -1, -1, -1
    scene = "unknown"
    horizontal_margin = _HORIZONTAL_MARGIN 

    # --- Calculate Dynamic Default Width (Helper for Horizontal Fallback) ---
    def calculate_dynamic_default_width(start_x: int, direction: str) -> int:
        # If right , the availaible space is from right of the page - start point of x - margin
        if direction == 'right':
             available_space = page_width - start_x - horizontal_margin
        # If left , then start point of the x - margin
        else: # direction == 'left'
             available_space = start_x - horizontal_margin

        dynamic_w = line_height * default_width_factor_of_height
        # Cap width by absolute max, relative max based on available space (avoid space太大)
        capped_w = min(dynamic_w, max_absolute_default_width, max(0, available_space) * max_relative_default_width_factor)
        
        # Ensure width is at least the minimum required signature width (avoid space太小)
        final_w = max(capped_w, min_sig_width)
        if enable_debug_prints:
             st.info(f"    [calculate_dynamic_default_width] StartX:{start_x} Dir:{direction} Avail:{available_space}")
            #  logging.info(f"    [calculate_dynamic_default_width] StartX:{start_x} Dir:{direction} Avail:{available_space}")
             st.info(f"    LineH:{line_height} * Factor:{default_width_factor_of_height:.1f} = {dynamic_w:.0f}")
            #  logging.info(f"    LineH:{line_height} * Factor:{default_width_factor_of_height:.1f} = {dynamic_w:.0f}")
             st.info(f"    MaxAbs:{max_absolute_default_width}, MaxRel:{max_relative_default_width_factor:.1f} * Avail:{max(0, available_space)} = {max(0, available_space) * max_relative_default_width_factor:.0f}")
            #  logging.info(f"    MaxAbs:{max_absolute_default_width}, MaxRel:{max_relative_default_width_factor:.1f} * Avail:{max(0, available_space)} = {max(0, available_space) * max_relative_default_width_factor:.0f}")
             st.info(f"    Capped W: {capped_w:.0f}, Final W (>=Min {min_sig_width}): {final_w:.0f}")
            #  logging.info(f"    Capped W: {capped_w:.0f}, Final W (>=Min {min_sig_width}): {final_w:.0f}")
        return int(final_w) # Return the width that can use 

    # --- Scene Detection ---
    is_below_case = False
    label_stripped = label_text.strip()

    # 1. Check for "Below" Scenario
    # Condition: Label ends with a keyword AND (it's short OR no close token to the right)
    # 蠻多格子類型的最後一個字會是蓋章，簽名，簽章等
    label_ends_with_keyword = any(label_stripped.endswith(kw) for kw in BELOW_LABEL_KEYWORDS)
    if label_ends_with_keyword:
        # Check if there's NOT a close token immediately to the right
        # 檢查最短距離旁邊有沒有字, 有字多數is not格子，一般文字
        # Use a shorter distance check for this specific purpose
        short_check_dist = max(50, int(line_height * 1.5))
        nearest_right_short_dist_check = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=short_check_dist,
            y_tolerance_factor=0.6, 
            ignore_chars=":： ",
            debug_mode=enable_debug_prints
        )
        # If there is no word nearby then assume signature area is below
        if nearest_right_short_dist_check is None:
            is_below_case = True
            scene = "below_area"
            if enable_debug_prints: st.info(f"scenario: {scene} (Keyword suffix and no close right token within {short_check_dist}px)")
            # logging.info(f"scenario: {scene} (Keyword suffix and no close right token within {short_check_dist}px)" )
        elif enable_debug_prints:
             # explain why it wasn't treated as 'below_area' even with keyword match
             nr_token = nearest_right_short_dist_check
             st.info(f"  (Keyword suffix matched, but found close right token '{nr_token['text']}' at {nr_token['bbox']}, not treating as 'below')")
            #  logging.info(f"  (Keyword suffix matched, but found close right token '{nr_token['text']}' at {nr_token['bbox']}, not treating as 'below')")

    # --- Inference Logic based on Scene ---
    # If area is detected as below area
    if is_below_case:
        # --- Vertical Position ---
        sig_ymin = ly_max + vertical_margin_below
        # Estimate height based on label height (Option A)
        estimated_height = int(line_height * signature_area_height_factor_below)
        # Ensure minimum sensible height for the signature area
        estimated_height = max(estimated_height, _MIN_INFERRED_HEIGHT * 1.5) # Slightly larger min for below?
        sig_ymax = sig_ymin + estimated_height
        if enable_debug_prints: st.info(f"  '{scene}' Vertical: y_min={sig_ymin}, y_max={sig_ymax} (Est. H: {estimated_height})")
        # --- Horizontal Position ---
        sig_xmin = lx_min
        sig_xmax = lx_max
        # Ensure minimum width based on label height or absolute min
        min_req_width = max(min_sig_width, int(line_height * min_width_factor_below))
        current_width = sig_xmax - sig_xmin

        # If the width is too narrow, adjust the width
        if current_width < min_req_width:
            needed_expansion = min_req_width - current_width
            # Expand somewhat centered, respecting page boundaries
            expand_left = needed_expansion // 2
            # Calculate potential new xmin and check boundary
            potential_xmin = sig_xmin - expand_left
            actual_xmin = max(0, potential_xmin)
            actual_left_expansion = sig_xmin - actual_xmin # How much it actually expanded left

            # Calculate right expansion needed based on actual left expansion
            expand_right = needed_expansion - actual_left_expansion
            potential_xmax = sig_xmax + expand_right
            actual_xmax = min(page_width, potential_xmax)

            sig_xmin = actual_xmin
            sig_xmax = actual_xmax

            # Re-check width after boundary adjustments
            final_width = sig_xmax - sig_xmin
            if final_width < min_sig_width * 0.9: # Allow slight tolerance
                 if enable_debug_prints: st.warning(f"Warning: '{scene}' Expanded Width ({final_width}) still narrow than  ~{min_sig_width}。")
                #  logging.warning(f"Warning: '{scene}' Expanded Width ({final_width}) still narrow than  ~{min_sig_width}。"
            scene += "_expand_width"
            if enable_debug_prints: st.info(f"  '{scene}' Width expanded from {current_width} to -> {final_width} (Target: {min_req_width}) => xmin={sig_xmin}, xmax={sig_xmax}")
            # logging.info(f"  '{scene}' Width expanded from {current_width} to -> {final_width} (Target: {min_req_width}) => xmin={sig_xmin}, xmax={sig_xmax}")
        else:
             if enable_debug_prints: st.info(f"  '{scene}' Horizontal: x_min={sig_xmin}, x_max={sig_xmax} (Width {current_width} >= {min_req_width})")
            #  logging.info(f"  '{scene}' Horizontal: x_min={sig_xmin}, x_max={sig_xmax} (Width {current_width} >= {min_req_width})")


    # 2. Check for Horizontal Scenarios 
    # Scenario of right colon 簽名:______, 簽章:     
    elif label_stripped.endswith(':') or label_stripped.endswith('：'):
        scene = "right_colon"
        if enable_debug_prints: st.info(f"scenario: {scene}")
        
        # avoid overlap with the label 
        sig_xmin = lx_max + horizontal_margin
        # --- Vertical position: Center align with label ---
        # Calculate height
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)  # Ensure min height
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        # --- Horizontal extent ---
        
        nearest_right_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        # Check whether is 簽名: _______ (簽章) 文字間
        is_newline_paren_case = False
        if nearest_right_token:
            nr_text = nearest_right_token["text"]
            # check : 附近時不是括弧
            if NEWLINE_PAREN_PATTERN.search(nr_text):
                is_newline_paren_case = True
                scene = "right_colon_with_text"
                if enable_debug_prints: st.info(f"  This is '{scene}' scenario: '{nr_text}'")

                # Use extract_signature_blank 
                if image is not None:
                  # define search area  from the right of the label to the right of the token
                    search_xmin = lx_max
                    search_xmax = nearest_right_token["bbox"][2]  
                    search_ymin = sig_ymin
                    search_ymax = sig_ymax
                    search_bbox = (search_xmin, search_ymin, search_xmax, search_ymax)

                    if enable_debug_prints:
                        st.info(f"  Using adaptiveThreshold search area: {search_bbox}")
                        # logging.info(f"  Using adaptiveThreshold search area: {search_bbox}")

                    blank_bboxes = extract_signature_blank(
                        image=image,
                        bbox=search_bbox,
                        blank_width_factor=1.0,
                        min_blank_width=min_sig_width
                    )

                    if blank_bboxes and len(blank_bboxes) > 0:
                        blank_xmin, blank_ymin, blank_xmax, blank_ymax = blank_bboxes[0]
                        # use blank boundingboxes coordinate
                        sig_xmin = blank_xmin
                        sig_xmax = blank_xmax
                        sig_ymin = blank_ymin
                        sig_ymax = blank_ymax
                        if enable_debug_prints:
                            st.info(f"  adaptiveThreshold for the blank area: ({sig_xmin}, {sig_ymin}, {sig_xmax}, {sig_ymax})")
                    else:
                        if enable_debug_prints:
                            st.info(f"  adaptiveThreshold failed，Using default scenario")
                            # logging.info(f"  adaptiveThreshold failed，Using default scenario")
                        is_newline_paren_case = False
                        scene = "right_colon"
                else:
                    if enable_debug_prints:
                        st.info(f"  Unable to use the image，adaptiveThreshold failed，Using default scenario")
                        # logging.info(f"  Unable to use the image，adaptiveThreshold failed，Using default scenario")
                    is_newline_paren_case = False
                    scene = "right_colon"

        if is_newline_paren_case and sig_xmin != -1 and sig_xmax != -1:
            # adaptiveThreshold success sig_xmin 和 sig_xmax
            if enable_debug_prints:
                st.info(f"  '{scene}' adaptiveThreshold generate blank area: sig_xmin={sig_xmin}, sig_xmax={sig_xmax}")
        elif nearest_right_token:
            nr_bbox = nearest_right_token["bbox"]
            # Adjust the width 
            sig_xmax = max(sig_xmin + min_sig_width // 2, nr_bbox[0] - horizontal_margin)
            if enable_debug_prints:
                st.info(f"  '{scene}' Found right token: '{nearest_right_token['text']}' => sig_xmax={sig_xmax}")
        else:
            # Token no found , dynamic set a width 
            dynamic_width = calculate_dynamic_default_width(sig_xmin, 'right')
            sig_xmax = min(sig_xmin + dynamic_width, page_width - horizontal_margin)
            if enable_debug_prints:
                st.info(f"  '{scene}' No right token found, using dynamic width {dynamic_width} => sig_xmax={sig_xmax}")
                
    # Scenario
    #.    _______(簽名)
    elif label_stripped.startswith('(') and label_stripped.endswith(')'):
        scene = "left_paren"
        if enable_debug_prints: st.write(f"Scenario: {scene}")
        # --- Horizontal Position ---
        sig_xmax = lx_min - horizontal_margin
        # --- Vertical position: Center align with label ---
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        # --- Horizontal extent ---
        nearest_left_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='left',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        if nearest_left_token:
             nl_bbox = nearest_left_token["bbox"]
             # Infer xmin based on nearest token, ensure minimum width gap
             sig_xmin = min(sig_xmax - min_sig_width // 2, nl_bbox[2] + horizontal_margin)
             if enable_debug_prints: st.info(f"  '{scene}' Found left token: '{nearest_left_token['text']}' => sig_xmin={sig_xmin}")
            #  logging.info(f"  '{scene}' Found left token: '{nearest_left_token['text']}' => sig_xmin={sig_xmin}")
        else:
             # No token found, use dynamic default width
             dynamic_width = calculate_dynamic_default_width(sig_xmax, 'left')
             sig_xmin = max(sig_xmax - dynamic_width, horizontal_margin)
             if enable_debug_prints: st.info(f"  '{scene}' No left token found, using dynamic width {dynamic_width} => sig_xmin={sig_xmin}")
            #  logging.info(f"  '{scene}' No left token found, using dynamic width {dynamic_width} => sig_xmin={sig_xmin}")

    else: # If not match scenario above , all use right colon method
        scene = "fallback_right"
        if enable_debug_prints: st.info(f"Scenario: {scene} (Defaulting to right of label)")
        # logging.info(f"Scenario: {scene} (Defaulting to right of label)")
        sig_xmin = lx_max + horizontal_margin
        # --- Vertical position: Center align with label ---
        target_center_y = (ly_min + ly_max) / 2
        inferred_height = int(line_height * signature_area_height_factor)
        inferred_height = max(inferred_height, _MIN_INFERRED_HEIGHT)
        sig_ymin = int(target_center_y - inferred_height / 2)
        sig_ymax = sig_ymin + inferred_height
        # --- Horizontal extent ---
        nearest_right_token = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=nearest_token_max_dist,
            y_tolerance_factor=nearest_token_y_factor,
            debug_mode=enable_debug_prints
        )
        if nearest_right_token:
             nr_bbox = nearest_right_token["bbox"]
             sig_xmax = max(sig_xmin + min_sig_width // 2, nr_bbox[0] - horizontal_margin)
             if enable_debug_prints: st.write(f"  '{scene}' Found right token: '{nearest_right_token['text']}' => sig_xmax={sig_xmax}")
            #  logging.info(f"  '{scene}' Found right token: '{nearest_right_token['text']}' => sig_xmax={sig_xmax}")
        else:
             dynamic_width = calculate_dynamic_default_width(sig_xmin, 'right')
             sig_xmax = min(sig_xmin + dynamic_width, page_width - horizontal_margin)
             if enable_debug_prints: st.write(f"  '{scene}' No right token found, using dynamic width {dynamic_width} => sig_xmax={sig_xmax}")
            #  logging.info(f"  '{scene}' No right token found, using dynamic width {dynamic_width} => sig_xmax={sig_xmax}")

    # --- Final Checks 
    if sig_xmin == -1 or sig_xmax == -1 or sig_ymin == -1 or sig_ymax == -1:
        if enable_debug_prints: st.error(f"Error: Cannot calculate all the bounding box (xmin={sig_xmin}, ymin={sig_ymin}, xmax={sig_xmax}, ymax={sig_ymax})。")
        # logging.error(f"Error: Cannot calculate all the bounding box (xmin={sig_xmin}, ymin={sig_ymin}, xmax={sig_xmax}, ymax={sig_ymax})。")
        return None

    # Final width check - ensure min_sig_width is met after all logic
    calculated_width = sig_xmax - sig_xmin
    if calculated_width < min_sig_width:
         if enable_debug_prints: st.warning(f"Warining: Final width ({calculated_width:.0f}) < {min_sig_width}。Fixing....。")
        #  logging.warning(f"Warining: Final width ({calculated_width:.0f}) < {min_sig_width}。Fixing....。")

         if scene.startswith("left"): # Expand left if it was a left-based rule
             sig_xmin = max(0, sig_xmax - min_sig_width)
         else: # Default expand right
             sig_xmax = min(page_width, sig_xmin + min_sig_width)
         # If still too narrow (page too small), it is what it is
         if sig_xmax - sig_xmin < min_sig_width * 0.8:
              if enable_debug_prints: st.write(f"  Width fixed ({sig_xmax - sig_xmin}) not enough")
              # logging.info(f"  Width fixed ({sig_xmax - sig_xmin}) not enough")
         else:
              scene += "_final_min_width_fix"
              if enable_debug_prints: st.write(f"  Width fixed: xmin={sig_xmin}, xmax={sig_xmax}")
              # logging.info(f"  Width fixed: xmin={sig_xmin}, xmax={sig_xmax}")


    # Clip final coordinates to page boundaries
    sig_xmin = max(0, sig_xmin)
    sig_ymin = max(0, sig_ymin)
    sig_xmax = min(page_width, sig_xmax)
    sig_ymax = min(page_height, sig_ymax)

    # Final height check
    if sig_ymax <= sig_ymin:
        if enable_debug_prints: st.error(f"Error: Height Calculated is invalid (ymax <= ymin)。ymin={sig_ymin}, ymax={sig_ymax}")
        # logging.error(f"Error: Height Calculated is invalid (ymax <= ymin)。ymin={sig_ymin}, ymax={sig_ymax}")
        
        # Ajusting height: Attempt a simple Y fallback based on label position
        sig_ymin = ly_max + 2 # Start just below label
        sig_ymax = sig_ymin + max(int(line_height * 1.5), _MIN_INFERRED_HEIGHT) 
        sig_ymin = max(0, sig_ymin)
        sig_ymax = min(page_height, sig_ymax)
        if sig_ymax <= sig_ymin:
             if enable_debug_prints: st.error("Error: Final height check. Cannot predict signature area")
            #  logging.error("Error: Final height check. Cannot predict signature area")
             return None 
        scene += "_y_final_fallback"
        if enable_debug_prints: st.warning(f"Warining: Final height check: ymin={sig_ymin}, ymax={sig_ymax}")
        # logging.warning(f"Warining: Final height check: ymin={sig_ymin}, ymax={sig_ymax}")


    # Ensure coordinates are integers before returning
    inferred_bbox = (int(sig_xmin), int(sig_ymin), int(sig_xmax), int(sig_ymax))
    # Check and Validate 
    if inferred_bbox[2] <= inferred_bbox[0] or inferred_bbox[3] <= inferred_bbox[1]:
         if enable_debug_prints: st.error(f"Error: Invalid Bounding box:  {inferred_bbox}")
        #  logging.error(f"Error: Invalid Bounding box:  {inferred_bbox}")
         return None

    if enable_debug_prints: st.write(f"Final BBox ('{scene}'): {inferred_bbox}")
    # logging.info(f"Final BBox ('{scene}'): {inferred_bbox}")
    return inferred_bbox, scene

def process_document_ai(file_path: str) -> Optional[documentai.Document]:
    global client, processor_name # Use global client and processor name

    if not os.path.exists(file_path):
        st.error(f"Error: File not exist {file_path}")
        return None

    try:
        # MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
             _, ext = os.path.splitext(file_path)
             ext = ext.lower()
             if ext == ".pdf": mime_type = "application/pdf"
             elif ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
             elif ext == ".png": mime_type = "image/png"
             elif ext == ".tiff" or ext == ".tif": mime_type = "image/tiff"
             else: raise ValueError(f"File type not sure: {file_path}")

        st.info(f"Handling : {file_path}, MIME type: {mime_type}")

        # Check supported MIME types
        supported_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg", "image/tiff"]
        if mime_type not in supported_types:
            raise ValueError(f"File Not Support: {mime_type} Please provide PDF, JPG, PNG, TIFF format")

        # Read file content
        with open(file_path, "rb") as file:
            content = file.read()

        # Prepare Document AI request
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        # Add ProcessOptions only if needed (e.g., for specific OCR versions or layout parsing)
        # process_options = documentai.ProcessOptions(...)
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
            # process_options=process_options # Uncomment if using process_options
            # skip_human_review=True # Set to True if you don't use Human-in-the-Loop
        )

        st.info("Using Document AI Process API...")
        result = documentai_client.process_document(request=request)
        st.success("Document AI is finished。")
        return result.document

    except Exception as e:
        st.error(f"Handling {file_path} Error: {e}")
        return None

def extract_and_infer_signature_areas(document: documentai.Document) -> Tuple[List[Dict], Dict]:
    results = []
    doc_ai_dimensions = {} # 存儲 DocAI 的頁面尺寸
    if not document.pages:
        st.warning("Warning: Document AI has no page details。")
        return results, doc_ai_dimensions

    page_tokens_map = {}
    # Pre-process all pages to get tokens and dimensions
    for i, page in enumerate(document.pages):
        if not page.dimension or not page.dimension.width or not page.dimension.height or page.dimension.width <= 0 or page.dimension.height <= 0:
            st.warning(f"Warning: Page {i} invalid. Skip")
            continue

        page_width_docai = int(page.dimension.width)
        page_height_docai = int(page.dimension.height)

        # Store DocAI dimensions for later use (e.g., scaling in visualization)
        doc_ai_dimensions[i] = {'width': page_width_docai, 'height': page_height_docai}

        # Extract tokens for the current page
        page_tokens_map[i] = {
            "tokens": get_page_tokens(page, page_width_docai, page_height_docai),
            "width": page_width_docai,
            "height": page_height_docai
        }

    if not document.entities:
        st.warning("Warning: Document AI has not label entities.")
        return results, doc_ai_dimensions

    target_entity_type = "signature_field"
    if enable_debug_prints:
        st.info(f"From {len(document.entities)} Entities searching '{target_entity_type}'...")

    processed_entity_count = 0
    min_sig_width_default = 50  # Provide a default value

    for entity in document.entities:
        if entity.type == target_entity_type:
            # Basic entity validation
            if not entity.page_anchor or not entity.page_anchor.page_refs:
                 st.warning(f"Warning: Entity '{entity.mention_text}' has no page_anchor or page_refs")
                 continue

            page_ref = entity.page_anchor.page_refs[0]
            page_num = page_ref.page # page_num is 0-based index

            # Check if page data was successfully processed earlier
            if page_num not in page_tokens_map:
              st.warning(f"Warning: The page {page_num} corresponding to entity '{entity.mention_text}' has not been processed (possibly due to invalid dimensions).")
              continue


            page_info = page_tokens_map[page_num]
            page_width_docai = page_info["width"]
            page_height_docai = page_info["height"]
            page_tokens = page_info["tokens"]

            # Check for bounding poly on the page reference(Actual Coordinate)
            if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
                st.warning(f"Warning: The page_ref for entity '{entity.mention_text}' (Page {page_num}) is missing bounding_poly.")
                continue

            # Store normalized vertices for visualization scaling
            label_bbox_normalized = page_ref.bounding_poly.normalized_vertices

            inference_result = infer_signature_area_bbox(
                label_entity=entity,
                page_tokens=page_tokens,
                page_width=page_width_docai,
                page_height=page_height_docai,
                min_sig_width=min_sig_width_default,  
                enable_debug_prints=enable_debug_prints 
            )

            inferred_area_bbox_pixel_docai = None # BBox relative to DocAI dimensions
            inferred_rule = "failed"  # Initial
            if inference_result:
                # Unpack result if inference was successful
                inferred_area_bbox_pixel_docai, inferred_rule = inference_result
                processed_entity_count += 1


            # Calculate label bbox again just for storing in results, if needed
            # Can be recalculated later from normalized vertices during visualization
            label_bbox_pixel_docai_for_storage = get_pixel_bbox(label_bbox_normalized, page_width_docai, page_height_docai)

            # Append result
            results.append({
                "entity_type": target_entity_type,
                "label_text": entity.mention_text,
                "label_bbox_normalized": label_bbox_normalized, 
                "label_bbox_pixel_docai": label_bbox_pixel_docai_for_storage,  # pixel邊界框
                "inferred_area_bbox_pixel_docai": inferred_area_bbox_pixel_docai, 
                "inferred_rule": inferred_rule,
                "confidence": entity.confidence,
                "page": page_num, 
                "min_sig_width": min_sig_width_default   # Minium width of signature area
            })
    if enable_debug_prints:
        st.info(f"Found and attempting to process {len([r for r in results if r['entity_type'] == target_entity_type])} '{target_entity_type}' entities.")
        st.info(f"Successfully inferred {processed_entity_count} signature regions.")
    return results, doc_ai_dimensions

# --- Main Processing Logic ---
if st.session_state.uploaded_file:
    st.text(f"已選擇文件：{st.session_state.uploaded_file.name}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("開始"):
            if file_source == "Upload File" and st.session_state.uploaded_file:
                file_bytes = st.session_state.uploaded_file.read()
                file_extension = os.path.splitext(st.session_state.uploaded_file.name)[1]
            elif file_source == "Use Default File" and st.session_state.uploaded_file:
                default_file_path = "pages/references/社會住宅包租代管第4期計畫民眾承租住宅申請書1120621.pdf"
                st.session_state.uploaded_file.seek(0) # Reset file pointer to the beginning
                file_bytes = st.session_state.uploaded_file.read()
                file_extension = os.path.splitext(default_file_path)[1]
            else:
                st.error("請先上傳文件或選擇預設文件。")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                file_path = tmp_file.name
            if file_source == "Upload File" and not (os.path.isfile(".env") or os.getenv("ENV") == "dev"):
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
            def visualize_results(file_path: str, processed_results: List[Dict], doc_ai_dimensions: Dict,
                      _HORIZONTAL_MARGIN = 5,
                      _MIN_INFERRED_HEIGHT = 15,
                      vertical_margin_below=10,
                      min_width_factor_below=3.0):
                """Visualize processed_results, recalculating coordinates based on the actual image dimensions for drawing."""
                if not os.path.exists(file_path):
                    print(f"Error: Visualization file does not exist {file_path}")
                    # logging.error(f"Error: Visualization file does not exist {file_path}")
                    return

                # Determine MIME type for loading
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    _, ext = os.path.splitext(file_path)
                    ext = ext.lower()
                    if ext == ".pdf": mime_type = "application/pdf"
                    elif ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                    elif ext == ".png": mime_type = "image/png"
                    elif ext == ".tiff" or ext == ".tif": mime_type = "image/tiff"
                    else:
                        print(f"Error: Unable to determine MIME type for visualization file: {file_path}")
                        # logging.error(f"Error: Unable to determine MIME type for visualization file: {file_path}")
                    return

                print(f"\nStarting visualization of file: {file_path} (type: {mime_type})")
                images_cv = []
                try: # Load file into OpenCV image(s)
                    if mime_type == "application/pdf":
                        # Convert PDF to list of PIL images, then to OpenCV format
                        # Adjust dpi and size as needed for desired resolution vs. memory usage
                        images_pil = convert_from_path(file_path, dpi=200, timeout=180, size=(None, 4000))
                        images_cv = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images_pil] #將 PIL Image轉為 NumPy array並從 RGB 轉為 BGR
                        st.success(f"Loaded {len(images_cv)} pages from PDF.")
                    elif mime_type in ["image/jpeg", "image/png", "image/jpg", "image/tiff"]:
                        img_cv = cv2.imread(file_path)
                        if img_cv is None: raise ValueError(f"Unable to read image file with OpenCV: {file_path}")
                        images_cv = [img_cv]
                        st.success(f"Image file loaded.")
                    else:
                        raise ValueError(f"Unsupported visualization file type: {mime_type}")
                except Exception as e:
                    st.error(f"Failed to load image file: {e}")
                    return
                if not images_cv:
                    st.error(f"Error: Failed to load any image pages from {file_path} for visualization.")
                    return

                for page_num, img_cv in enumerate(images_cv):
                    # Basic check on loaded image page
                    if img_cv is None or img_cv.size == 0:
                        st.warning(f"Warning: Image data for page {page_num} is invalid, skipping.")
                        continue

                    # Ensure image is in BGR format for cv2 drawing functions
                    try:
                        if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
                            if len(img_cv.shape) == 2: # Grayscale --> BGRA
                                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                            elif img_cv.shape[2] == 4: # RGBA --> BGRA
                                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR) # Assuming RGBA from PIL
                            else: # Other formats: Taking first 3 channels
                                img_cv = img_cv[:, :, :3]
                    # Final check after conversion attempts
                        if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
                            raise ValueError("Unable to convert image to 3-channel BGR format")
                    except Exception as convert_err:
                        st.error(f"Error: Unable to convert page {page_num} to BGR format: {convert_err}, skipping.")
                        continue

                    vis_height, vis_width, _ = img_cv.shape
                    st.info(f"--- Visualizing page {page_num} ---")
                    if enable_debug_prints:
                        st.info(f"  Actual image dimensions (after rendering/loading): {vis_width}x{vis_height}")
                    doc_page_width = None
                    doc_page_height = None
                    scale_x = 1.0
                    scale_y = 1.0
                    if page_num in doc_ai_dimensions:
                        doc_page_info = doc_ai_dimensions[page_num]
                        doc_page_width = doc_page_info.get('width')
                        doc_page_height = doc_page_info.get('height')
                        if doc_page_width and doc_page_height and doc_page_width > 0 and doc_page_height > 0: # If valid size
                            if enable_debug_prints:
                                st.info(f"  DocAI reported dimensions: {doc_page_width}x{doc_page_height}")
                            # logging.info(f"  DocAI reported dimensions: {doc_page_width}x{doc_page_height}")
                            # Check for significant size mismatch
                            if abs(doc_page_width - vis_width) > 5 or abs(doc_page_height - vis_height) > 5:
                                scale_x = vis_width / doc_page_width
                                scale_y = vis_height / doc_page_height
                                if enable_debug_prints:
                                    st.info(f"  !!!! Size mismatch !!!! Calculated scaling factors: X={scale_x:.3f}, Y={scale_y:.3f}")
                            else:
                                st.warning(f"  Dimensions match, scaling factor is 1.0")
                        else:
                            st.warning(f"Warning: DocAI dimension information for page {page_num} is invalid. Assuming scaling factor of 1.0.")
                            doc_page_width, doc_page_height = vis_width, vis_height 
                    else:
                        st.warning(f"Warning: No DocAI dimension information found for page {page_num}. Assuming scaling factor of 1.0.")
                        doc_page_width, doc_page_height = vis_width, vis_height
                    page_results = [res for res in processed_results if res.get("page") == page_num]
                    if not page_results:
                        st.info(f"Page {page_num} has no detection results to draw.")
                        # continue # Skip to next page if no results for this one

                    # Create a copy of the image to draw on
                    img_to_show = img_cv.copy()
                    for result_idx, result in enumerate(page_results):
                        if enable_debug_prints:
                            st.info(f"  Processing result {result_idx+1}/{len(page_results)}: Label='{result.get('label_text', 'N/A')}'")
                        entity_type = result.get("entity_type", "unknown_label")
                        inferred_rule = result.get("inferred_rule", "N/A")
                        label_normalized_vertices = result.get("label_bbox_normalized")

                        # --- Calculate and Draw Label BBox (Red) bounding box that predict by DocumentAI---
                        label_bbox_vis = None 
                        if label_normalized_vertices:
                            # Calculate pixel bbox based on VISUAL image dimensions
                            label_bbox_vis = get_pixel_bbox(label_normalized_vertices, vis_width, vis_height)

                        if label_bbox_vis and len(label_bbox_vis) == 4:
                            lx_vis, ly_vis, lx2_vis, ly2_vis = label_bbox_vis
                            # Ensure coordinates are valid before drawing
                            if lx_vis < lx2_vis and ly_vis < ly2_vis:
                                try:
                                    # Draw red rectangle for the label
                                    cv2.rectangle(img_to_show, (lx_vis, ly_vis), (lx2_vis, ly2_vis), (0, 0, 255), 2) # BGR Red 255
                                    # Add text label above the box
                                    text_y = ly_vis - 10 if ly_vis > 15 else ly2_vis + 20
                                    label_text_simple = f"{entity_type}" 
                                    cv2.putText(img_to_show, label_text_simple, (lx_vis, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                except Exception as draw_err:
                                    st.error(f"Error: Failed to draw Label '{entity_type}': {draw_err}, BBox={label_bbox_vis}")
                            else:
                                st.warning(f"Warning: Visualization BBox for Label '{entity_type}' is invalid ({label_bbox_vis}), cannot draw.")
                                #  logging.warning(f"Warning: Visualization BBox for Label '{entity_type}' is invalid ({label_bbox_vis}), cannot draw.")
                                label_bbox_vis = None 
                        else:
                            st.warning(f"Warning: Unable to calculate Label BBox for '{entity_type}' for drawing.")
                            # logging.warning(f"Warning: Unable to calculate Label BBox for '{entity_type}' for drawing.")
                            continue 
                        label_h_vis = ly2_vis - ly_vis if label_bbox_vis else 20 # Visual height of label, default 20 avoid error
                        # --- Calculate Inferred BBox for drawing (Green) bounding box that from post processing) ---
                        inferred_bbox_vis = None # Initial
                        inferred_bbox_docai = result.get("inferred_area_bbox_pixel_docai") # Get inferred box (DocAI coords)

                        if inferred_bbox_docai and len(inferred_bbox_docai) == 4:
                            ix_docai, iy_docai, ix2_docai, iy2_docai = inferred_bbox_docai
                            inf_w_docai = ix2_docai - ix_docai
                            inf_h_docai = iy2_docai - iy_docai

                            # Check if inferred DocAI box is valid before scaling
                            if inf_w_docai > 0 and inf_h_docai > 0:
                                try:
                                    # 1. Calculate scaled width/height for visualization
                                    new_inf_w_vis = inf_w_docai * scale_x
                                    new_inf_h_vis = inf_h_docai * scale_y
                                    # Enforce minimum visual height
                                    new_inf_h_vis = max(new_inf_h_vis, _MIN_INFERRED_HEIGHT)

                                    # --- 2. Calculate Visual Position based on Inference Rule ---
                                    ix_vis, iy_vis, ix2_vis, iy2_vis = -1,-1,-1,-1 # Initialize visual coords

                                    # Get visual label coords again (needed for all rules)
                                    lx_vis, ly_vis, lx2_vis, ly2_vis = label_bbox_vis

                                    if inferred_rule.startswith("below"):
                                        # Vertical position based on VISUAL label bottom + scaled margin
                                        scaled_margin_y = int(vertical_margin_below * scale_y)
                                        iy_vis = ly2_vis + scaled_margin_y
                                        iy2_vis = iy_vis + int(new_inf_h_vis)
                                        # Horizontal position: Scale DocAI coords or align with visual label
                                        # Scaling DocAI coords is often more reliable if expansion logic was complex
                                        ix_vis_scaled = ix_docai * scale_x
                                        ix2_vis_scaled = ix2_docai * scale_x
                                        current_width_vis = ix2_vis_scaled - ix_vis_scaled

                                        # Optional: Re-apply minimum width check based on visual dimensions
                                        min_sig_width=50
                                        min_req_width_vis = max(min_sig_width * scale_x, int(label_h_vis * min_width_factor_below))
                                        if current_width_vis < min_req_width_vis * 0.9:
                                            st.info(f"    '{inferred_rule}': Visual width {current_width_vis:.0f} < target {min_req_width_vis:.0f}. Expanding visually.")
                                            # Center expansion based on visual label center
                                            center_x_vis = (lx_vis + lx2_vis) / 2
                                            half_width = min_req_width_vis / 2
                                            ix_vis = center_x_vis - half_width
                                            ix2_vis = center_x_vis + half_width
                                        else:
                                            ix_vis = ix_vis_scaled
                                            ix2_vis = ix2_vis_scaled


                                    elif inferred_rule.startswith("right") or inferred_rule.startswith("fallback"):
                                        # Horizontal position based on VISUAL label right + scaled margin
                                        scaled_margin_x = int(_HORIZONTAL_MARGIN * scale_x)
                                        ix_vis = lx2_vis + scaled_margin_x
                                        ix2_vis = ix_vis + int(new_inf_w_vis)
                                        # Vertical position: center align with VISUAL label's vertical center
                                        center_y_vis = (ly_vis + ly2_vis) / 2
                                        iy_vis = center_y_vis - int(new_inf_h_vis / 2)
                                        iy2_vis = iy_vis + int(new_inf_h_vis)

                                    elif inferred_rule.startswith("left"):
                                        # Horizontal position based on VISUAL label left - scaled margin
                                        scaled_margin_x = int(_HORIZONTAL_MARGIN * scale_x)
                                        ix2_vis = lx_vis - scaled_margin_x
                                        ix_vis = ix2_vis - int(new_inf_w_vis)
                                        # Vertical position: center align with VISUAL label's vertical center
                                        center_y_vis = (ly_vis + ly2_vis) / 2
                                        iy_vis = center_y_vis - int(new_inf_h_vis / 2)
                                        iy2_vis = iy_vis + int(new_inf_h_vis)

                                    else: # Unknown or failed rule - attempt fallback drawing to the right
                                        st.info(f"    Processing unknown or failed rule '{inferred_rule}', attempting to draw on the right.")
                                        #  logging.info(f"    Processing unknown or failed rule '{inferred_rule}', attempting to draw on the right.")
                                        scaled_margin_x = int(_HORIZONTAL_MARGIN * scale_x)
                                        ix_vis = lx2_vis + scaled_margin_x
                                        ix2_vis = ix_vis + int(new_inf_w_vis) # Use scaled width
                                        center_y_vis = (ly_vis + ly2_vis) / 2
                                        iy_vis = center_y_vis - int(new_inf_h_vis / 2) # Use scaled height
                                        iy2_vis = iy_vis + int(new_inf_h_vis)

                                    # 3. Clip coordinates to visual page boundaries
                                    ix_vis = int(max(0, ix_vis))
                                    iy_vis = int(max(0, iy_vis))
                                    ix2_vis = int(min(vis_width, ix2_vis))
                                    iy2_vis = int(min(vis_height, iy2_vis))

                                    # 4. Final check for visual validity
                                    if ix2_vis > ix_vis and iy2_vis > iy_vis:
                                        inferred_bbox_vis = (ix_vis, iy_vis, ix2_vis, iy2_vis)
                                    else:
                                        st.info(f"Warning: Calculated Inferred BBox for drawing is invalid (Rule: {inferred_rule}). Coords: {(ix_vis, iy_vis, ix2_vis, iy2_vis)}")
                                        #  logging.warning(f"Warning: Calculated Inferred BBox for drawing is invalid (Rule: {inferred_rule}). Coords: {(ix_vis, iy_vis, ix2_vis, iy2_vis)}")

                                except Exception as calc_err:
                                    st.info(f"Error: Failed to calculate Inferred BBox for drawing (Rule: {inferred_rule}): {calc_err}")
                                    inferred_bbox_vis = None
                            else:
                                st.info(f"Warning: Inferred DocAI BBox is invalid ({inferred_bbox_docai}), unable to scale for visualization.")
                                #  logging.warning(f"Warning: Inferred DocAI BBox is invalid ({inferred_bbox_docai}), unable to scale for visualization.")

                        # --- Draw Inferred BBox (Green) Bounding box that from post processing ---
                        if inferred_bbox_vis:
                            ix, iy, ix2, iy2 = inferred_bbox_vis
                            try:
                                # Draw green rectangle for the inferred area
                                cv2.rectangle(img_to_show, (ix, iy), (ix2, iy2), (0, 255, 0), 2) # BGR Green
                                # Add text label for the inferred area
                                text_y = iy - 10 if iy > 15 else iy2 + 20
                                rule_str = str(inferred_rule).replace("_", " ") # Clean up rule string
                                inferred_text_simple = f"SignArea ({rule_str})"
                                # Adjust text position if it overlaps too much with label text
                                if abs(iy - ly_vis) < 30 and ix < lx2_vis + 50: text_y = iy2 + 20
                                    # Make font slightly smaller
                                cv2.putText(img_to_show, inferred_text_simple, (ix, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2) # Darker Green
                                # print(f"    Drawing Inferred (green) @ ({ix},{iy})-({ix2},{iy2}) (Rule: {rule_str})")
                            except Exception as draw_err:
                                st.info(f"Error: Failed to draw Inferred box/text: {draw_err}, BBox={inferred_bbox_vis}")

                        elif label_bbox_vis: # Only draw 'failed' if label was drawn but inference failed/invalid
                            try:
                                # Draw text indicating failure near the label
                                lx2_vis, ly_vis, _, _ = label_bbox_vis # Get label position again
                                cv2.putText(img_to_show, 'Area Infer Failed', (lx2_vis + 5, ly_vis + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1) # Red text
                                st.info(f"    Marked inference failure (Rule: {inferred_rule})")
                            except Exception as draw_err:
                                st.info(f"Error: Failed to draw 'Area Infer Failed' text: {draw_err}")

                    # --- Display image page using Streamlit ---
                    try:
                        st.image(img_to_show, caption=f"Page {page_num + 1} - Signature Area Inference (DocAI Label: Red, Post-Processing: Green)")
                        st.info(f"--- Visualization of page {page_num} completed ---")
                    except Exception as plot_err:
                        st.error(f"Error: Failed to display page {page_num} with Streamlit: {plot_err}")

            # --- Process and Display ---
            with st.spinner("ANALYSING..."):
                document = process_document_ai(file_path)
                if document:
                    processed_results, doc_ai_dimensions = extract_and_infer_signature_areas(document)
                    boxes = processed_results  # Already in the required format
                    st.session_state.boxes = boxes
                    st.subheader("Visualization")
                    visualize_results(file_path, boxes, doc_ai_dimensions)
                    if boxes:
                        st.success("SUCCESSFULLY DETECTED SIGNATURE FIELDS!")
                        st.text("Bounding Boxes:")
                        st.write(boxes)
                    else:
                        st.warning("沒找到簽名欄位，或者處理過程中出了點問題。")
                else:
                    st.error("Failed to process document with Document AI.")

    with col2:
        if st.button("Clear"):
            st.session_state.uploaded_file = None
            st.session_state.boxes = None
            st.session_state.gcs_path = None
            st.session_state.uploader_key = f"doc_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.file_source = "Upload File" # Reset file source to default
            st.rerun()
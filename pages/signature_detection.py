import pytz
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
import re
import time

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
with st.expander("IDP Signature Field Detection (trial version)"):
    st.write("""
**Summary**：  
此系統由 Kdan Infra 開發，通過自動檢測 PDF 和圖像（JPG、PNG）中的簽名欄位來簡化文件處理流程。
系統利用 Google Cloud 的 Document AI 提取結構化數據並推斷簽名區域，提升文件處理效率。

**解決的痛點**：  
- :tired_face: **手動審查耗時**：傳統方法需人工定位簽名欄位，拖慢工作流程。  
- :negative_squared_cross_mark: **檢測不一致**：文件佈局多樣化導致簽名欄位識別難以標準化。  
- :dancers: **擴展性問題**：手動處理難以應對大量文件。

**Process**：  
1. **上傳**：用戶上傳 PDF 或圖像文件。  
2. **OCR 與實體提取**：Document AI 處理文件，識別簽名欄位實體。  
3. **簽名區域推斷**：根據標籤文字與周圍標記，自定義邏輯推斷簽名位置。  
4. **視覺化**：顯示結果，標籤框為紅色，簽名區域為綠色。

**Future**：  
這是MVP版本。後續將引入佈局感知的多模態模型（結合文字、圖像與空間數據），提升準確性並處理複雜文件結構。這部分會需要花更多時間及計算資源來完成，未來會訓練本地模型來取代雲端模型。

**Pricing**:  
USD\$ 30 / 1000 pages = USD/\$ 0.03/page
 
""")

# Sidebar for optional GCP JSON key upload
st.sidebar.subheader(":file_folder: File Source :file_folder:")
file_source = st.sidebar.selectbox("Select File Source", ["Upload File", "Use Default File"])
st.sidebar.divider()
st.sidebar.subheader(":secret: :key: Upload GCP JSON Key (Optional)")
st.sidebar.text("Default credentials are in use. Upload your own JSON key to override.")
sa_key = st.sidebar.file_uploader("Upload JSON key", type=["json"], key="sa_key")
st.sidebar.divider()
enable_debug_prints = st.sidebar.checkbox("Enable Debug Mode", value=False)
if sa_key:
    custom_key = json.load(sa_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(custom_key, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
    st.sidebar.success(":white_check_mark: JSON KEY HAS BEEN UPLOADED (Overriding default)")

# --- Initialize Google Cloud Clients ---
try:
    documentai_client = documentai.DocumentProcessorServiceClient()
    storage_client = storage.Client()
except Exception as e:
    st.error(f":x:Failed to initialize GCP clients. Please check credentials: {e}")
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
if 'doc_dimensions' not in st.session_state:
    st.session_state.doc_dimensions = None
if 'current_processed_results' not in st.session_state:
    st.session_state.current_processed_results = None
if 'current_file_path' not in st.session_state:
    st.session_state.current_file_path = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_value' not in st.session_state:
    st.session_state.feedback_value = None
if 'feedback_comment' not in st.session_state:
    st.session_state.feedback_comment = ""

# --- File Upload UI ---
st.subheader(":file_folder: Upload Target File")

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
        st.info(f":pushpin: Default file selected: {os.path.basename(default_file_path)}")
        st.session_state.uploaded_file = open(default_file_path, "rb") # Open default file in binary read mode
    else:
        st.error(f":x: Default file path error: {default_file_path}")
        st.stop()

def get_pixel_bbox(normalized_vertices: List[documentai.NormalizedVertex], page_width: int, page_height: int) -> Optional[Tuple[int, int, int, int]]:
    if not normalized_vertices:
        st.warning(f":pushpin: Warning : Empty normalized_vertices。")
        return None

    try:
        # Filter out None coordinates and ensure they are within [0, 1]
        x_coords = [v.x for v in normalized_vertices if v.x is not None and 0 <= v.x <= 1]
        y_coords = [v.y for v in normalized_vertices if v.y is not None and 0 <= v.y <= 1]

        if not x_coords or not y_coords:
             st.warning(f":pushpin: Warning: unable extract the x, y coordinate from vertices: {normalized_vertices}")
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
             st.warning(f":pushpin: Warning: Invalid coordinates after calculation: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
             return None

        # Check for invalid box dimensions (max <= min)
        if xmax <= xmin or ymax <= ymin:
             # Check if there is negative, allow zero width/height initially, fix later if needed
             if xmax < xmin or ymax < ymin:
                 st.warning(f":pushpin: Warning: Invalid bounding box after calculation (max < min): xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
                 return None
             # else: # Handle zero width/height cases if necessary, often downstream logic handles this
             #     print(f"Log: BBox is a line or dot: xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}")
             #     xmax = max(xmax, xmin + 1)
             #     ymax = max(ymax, ymin + 1)
        return int(xmin), int(ymin), int(xmax), int(ymax)

    except Exception as e:
        st.error(f":x: Error: Calculation get_pixel_bbox : {e}, vertices: {normalized_vertices}")
        return None

## TODO
def get_page_tokens(page: documentai.Document.Page, page_width: int, page_height: int) -> List[Dict]:
    """Extract tokens and pixel bounding box from all pages."""
    tokens = []
    if not page.tokens:
         st.warning(f":pushpin: Warning: Page {page.page_number} has not tokens(Empty Page)。") 
         return tokens

    for token in page.tokens:
        # Basic validation, if 少資料then skip
        if not token.layout or not token.layout.text_anchor or not token.layout.text_anchor.content: 
          continue
        text = token.layout.text_anchor.content
        if not token.layout.bounding_poly or not token.layout.bounding_poly.normalized_vertices: 
          continue
        
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

def calculate_dynamic_min_sig_width(page_tokens: List[Dict], page_width: int) -> int:
    """
    Dynamically calculate min_sig_width based on average token height on the page.
    根據頁面 token 的平均高度，動態計算簽名區域的最小寬度。
    
    Args:
        page_tokens: List of tokens on the page, each containing bbox information.
        page_width: Page width in pixels.
    
    Returns:
        Dynamically calculated min_sig_width in pixels.
    """
    if not page_tokens:
        return 50  # Default value

    # Calculate heights of all tokens
    heights = [token["bbox"][3] - token["bbox"][1] for token in page_tokens]
    avg_height = sum(heights) / len(heights) if heights else 20  # Avoid division by zero

    # Assume signature width is 3 times the average token height
    # 假設簽名寬度是平均高度的 3 倍
    dynamic_min_sig_width = int(avg_height * 3)

    # Set upper and lower limits based on page width
    # 根據頁面寬度設置上下限
    min_limit = 30  # Minimum 30 pixels
    max_limit = min(150, page_width * 0.1)  # Maximum 150 pixels or 10% of page width
    dynamic_min_sig_width = max(min_limit, min(dynamic_min_sig_width, max_limit))

    if enable_debug_prints:
        st.info(f"Dynamic min_sig_width: {dynamic_min_sig_width} (avg_height={avg_height:.1f}, page_width={page_width})")
    return dynamic_min_sig_width

def calculate_dynamic_y_tolerance(page_tokens: List[Dict]) -> float:
    """
    Dynamically calculate y_tolerance_factor based on vertical distribution of tokens on the page.
    根據頁面 token 的垂直分佈，動態計算垂直容差因子。
    
    Args:
        page_tokens: List of tokens on the page, each containing bbox information.
    
    Returns:
        Dynamically calculated y_tolerance_factor.
    """
    if not page_tokens:
        return 0.7  # Default value

    # Calculate heights of all tokens
    # 計算所有 token 的高度
    heights = [token["bbox"][3] - token["bbox"][1] for token in page_tokens]
    avg_height = sum(heights) / len(heights) if heights else 20

    # Calculate and sort vertical center points
    # 計算垂直中心點並排序
    y_centers = sorted([(token["bbox"][1] + token["bbox"][3]) / 2 for token in page_tokens])
    y_diffs = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
    avg_y_diff = sum(y_diffs) / len(y_diffs) if y_diffs else avg_height

    # Calculate y_tolerance_factor: ratio of line spacing to average height
    # 計算 y_tolerance_factor：行間距與平均高度的比值
    y_tolerance_factor = avg_y_diff / avg_height
    y_tolerance_factor = max(0.5, min(1.0, y_tolerance_factor))  # Limit the range

    if enable_debug_prints:
        st.info(f"Dynamic y_tolerance_factor: {y_tolerance_factor:.2f} (avg_y_diff={avg_y_diff:.1f}, avg_height={avg_height:.1f})")
    return y_tolerance_factor

def calculate_dynamic_max_horizontal_dist(page_tokens: List[Dict], page_width: int) -> int:
    """
    Dynamically calculate max_horizontal_dist based on page width and horizontal distribution of tokens.
    根據頁面寬度和 token 的水平分佈，動態計算最大水平距離。
    
    Args:
        page_tokens: List of tokens on the page, each containing bbox information.
        page_width: Page width in pixels.
    
    Returns:
        Dynamically calculated max_horizontal_dist in pixels.
    """
    if not page_tokens:
        return 800  # Default value

    # Calculate and sort horizontal center points of tokens
    # 計算 token 的水平中心點並排序
    x_centers = sorted([(token["bbox"][0] + token["bbox"][2]) / 2 for token in page_tokens])
    x_diffs = [x_centers[i+1] - x_centers[i] for i in range(len(x_centers)-1)]
    avg_x_diff = sum(x_diffs) / len(x_diffs) if x_diffs else page_width * 0.1

    # 取頁面寬度的 50% 和平均水平間距的 3 倍中的較小值
    max_dist = min(page_width * 0.5, avg_x_diff * 3)
    max_dist = max(200, max_dist)  # 最小 200 像素

    if enable_debug_prints:
        st.info(f"Dynamic max_horizontal_dist: {max_dist} (avg_x_diff={avg_x_diff:.1f}, page_width={page_width})")
    return int(max_dist)

def calculate_dynamic_height_factor(label_bbox: Tuple[int, int, int, int], page_tokens: List[Dict]) -> float:
    """
    Dynamically calculate signature_area_height_factor based on the label height and average token height on the page.
    根據標籤高度和頁面 token 的平均高度，動態計算簽名區域高度因子。
    
    Args:
        label_bbox: Label's pixel bounding box (xmin, ymin, xmax, ymax).
        page_tokens: List of tokens on the page.
    
    Returns:
        Dynamically calculated signature_area_height_factor.
    """
    if not page_tokens:
        return 0.8  # Default value

    # Calculate label height
    _, ly_min, _, ly_max = label_bbox
    label_height = ly_max - ly_min

    # Calculate average height of page tokens
    heights = [token["bbox"][3] - token["bbox"][1] for token in page_tokens]
    avg_height = sum(heights) / len(heights) if heights else 20

    # Adjust based on ratio of label height to average height
    # 根據標籤高度與平均高度的比值調整
    height_ratio = label_height / avg_height if avg_height > 0 else 1.0
    height_factor = 0.8 * height_ratio  # Base value 0.8, adjusted by ratio
    height_factor = max(0.5, min(1.5, height_factor))  # Limit the range

    if enable_debug_prints:
        st.info(f"Dynamic signature_area_height_factor: {height_factor:.2f} (label_height={label_height}, avg_height={avg_height:.1f})")
    return height_factor


## TODO
def find_nearest_token_on_line(
    target_bbox: Tuple[int, int, int, int],
    page_tokens: List[Dict], # Token list
    direction: str = 'right', 
    y_tolerance_factor: float = 0.7, # determine the token row
    max_horizontal_dist: int = 800, # default pixel 
    ignore_chars: str = ":： " # Characters to ignore when checking if token is empty
    ) -> Optional[Dict]:
    tx_min, ty_min, tx_max, ty_max = target_bbox
    target_cy = (ty_min + ty_max) / 2 # calculate center point
    target_height = ty_max - ty_min  # calculate height
    if target_height <= 0: target_height = 1 # Avoid division by zero

    nearest_token_info = None # To save the nearest token
    min_dist = float('inf')

    if enable_debug_prints:
        st.info(f":heavy_multiplication_x: [Debug find_nearest] Target: {target_bbox} CY: {target_cy:.1f}, H: {target_height}, Y-Tol(px): {target_height * y_tolerance_factor:.1f}")
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
            if enable_debug_prints: potential_matches.append({"text": tok_text, "bbox": tok_bbox, "cy": tok_cy, "on_line": False, "reason": "Empty"})
            continue

        # Check vertical alignment: absolute difference in centers < tolerance
        # Find on the same row
        y_distance = abs(tok_cy - target_cy)
        is_on_line = y_distance < (target_height * y_tolerance_factor)

        # For debugging: collect tokens somewhat close vertically
        if enable_debug_prints and y_distance < (target_height * 1.5):
             potential_matches.append({
                 "text": tok_text, "bbox": tok_bbox, "cy": tok_cy,
                 "on_line": is_on_line, "dist_y": y_distance, "reason": ""
                 })
        elif not is_on_line and enable_debug_prints:
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
                 if enable_debug_prints:
                    # Update reason for debug matches
                    for m in potential_matches:
                        if m["bbox"] == tok_bbox: m["reason"] = f"Potential Match (Dist: {current_distance_metric:.1f})"


    if enable_debug_prints:
        st.info(f":heavy_multiplication_x: [Debug find_nearest] Tokens near Target Y:")
        # logging.info(f"[Debug find_nearest] Tokens near Target Y:")
        potential_matches.sort(key=lambda x: x['bbox'][0]) # Sort by x-coordinate
        for p_match in potential_matches:
            status = ""
            if p_match.get("on_line"): status += "OnLine "
            if p_match.get("reason"): status += f"({p_match['reason']}) "
            st.info(f"  - Text: '{p_match['text']}', BBox: {p_match['bbox']}, CY: {p_match['cy']:.1f}, DistY: {p_match.get('dist_y', -1):.1f} {status}")
            # logging.info(f"  - Text: '{p_match['text']}', BBox: {p_match['bbox']}, CY: {p_match['cy']:.1f}, DistY: {p_match.get('dist_y', -1):.1f} {status}")
        if nearest_token_info:
            st.info(f":heavy_multiplication_x: [Debug find_nearest] Selected Nearest: '{nearest_token_info['text']}' BBox: {nearest_token_info['bbox']} DistMetric: {min_dist:.1f}")
            # logging.info(f"[Debug find_nearest] Selected Nearest: '{nearest_token_info['text']}' BBox: {nearest_token_info['bbox']} DistMetric: {min_dist:.1f}")
        else:
            st.info(f":heavy_multiplication_x: [Debug find_nearest] No suitable nearest token found in direction '{direction}' within {max_horizontal_dist}px.")
            # logging.info(f"[Debug find_nearest] No suitable nearest token found in direction '{direction}' within {max_horizontal_dist}px.")

    return nearest_token_info

def infer_signature_area_bbox(
    label_entity: documentai.Document.Entity, # Signature field label detected by Document AI
    page_tokens: List[Dict], # List of tokens on the page with their positions
    page_width: int, # Page width in pixels
    page_height: int, # Page height in pixels
    min_sig_width: int = 150,  # Minimum signature area width in pixels
    signature_area_height_factor: float = 0.8,  # Height factor for horizontal signature areas
    default_width_factor_of_height: float = 7.0,  # Width factor based on height for horizontal fallback cases
    max_absolute_default_width: int = 450,  # Maximum absolute width (in pixels) for horizontal fallback
    max_relative_default_width_factor: float = 0.6,  # Maximum width as a factor of available space
    nearest_token_max_dist: int = 1000, # Maximum distance to search for nearby tokens (in pixels)
    nearest_token_y_factor: float = 0.75, # Vertical tolerance factor for finding tokens on the same line

    # --- Parameters for 'below' case (when signature area is below the label) ---
    BELOW_LABEL_KEYWORDS = ["簽章", "簽名", "蓋章", "Signature"] , # Keywords indicating signature fields
    signature_area_height_factor_below: float = 1.8, # Height multiplier for signature areas below labels
    min_width_factor_below: float = 3.0 ,# Minimum width relative to label height for below case
    vertical_margin_below: int = 10, # Vertical margin between label bottom and signature area top

    # --- General parameters ---
    _HORIZONTAL_MARGIN = 5, # Horizontal margin in pixels
    _MIN_INFERRED_HEIGHT = 15, # Minimum inferred height for signature areas
    paren_sig_width_percent: float = 0.5, # Width percentage for parentheses case
    paren_sig_min_width: int = 50, # Minimum width for parentheses case
) -> Optional[Tuple[Tuple[int, int, int, int], str]]:
    """
    Infer the bounding box of a signature area based on the position of a label entity.
    根據標籤實體的位置推斷簽名區域的邊界框。
    
    This function analyzes the document structure around a detected signature label
    and uses multiple heuristics to determine the most likely location where a signature
    should be placed, handling various document layouts and formats.
    
    Args:
        label_entity: The signature field label entity detected by Document AI
        page_tokens: List of tokens on the page with their bounding boxes
        page_width/height: Dimensions of the page in pixels
        [Other parameters described in function definition]
    
    Returns:
        A tuple of ((xmin, ymin, xmax, ymax), scene_type) if successful,
        where scene_type describes which inference rule was used,
        or None if inference failed
    """
    

    # --- Extract the label's bounding box from Document AI entity ---
    if not label_entity.page_anchor or not label_entity.page_anchor.page_refs:
        if enable_debug_prints: st.info(f":pushpin: Warining: Label Entity '{label_entity.mention_text}' no page_refs。")
        # loggin.warning(f"Warining: Label Entity '{label_entity.mention_text}' no page_refs。")
        return None
    # Get first page reference containing label coordinate details
    page_ref = label_entity.page_anchor.page_refs[0]

    # Extract normalized vertices from the page reference
    if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
         if enable_debug_prints: 
          st.error(f":x: Error: Label Entity '{label_entity.mention_text}'  page_ref has no bounding_poly。")
        #  logging.error(f"Error: Label Entity '{label_entity.mention_text}'  page_ref has no bounding_poly。")
         return None
    # Convert normalized vertices to pixel coordinates
    label_bbox = get_pixel_bbox(page_ref.bounding_poly.normalized_vertices, page_width, page_height)
    if label_bbox is None:
        if enable_debug_prints: 
          st.error(f":x: Error: Error extract Label Entity '{label_entity.mention_text}' bounding box。")
        # logging.error(f"Error: Error extract Label Entity '{label_entity.mention_text}' bounding box。")
        return None

    # --- Initialize variables ---
    lx_min, ly_min, lx_max, ly_max = label_bbox
    label_text = label_entity.mention_text if label_entity.mention_text else "" # Handle empty mention_text
    # Calculate label height
    line_height = ly_max - ly_min
    if line_height <= 0: line_height = 20 # Avoid division by zero or negative height
    if enable_debug_prints: st.info(f"\Handling Label: '{label_text}', Boundingbox: {label_bbox}, H:{line_height}")
    # logging.info(f"\Handling Label: '{label_text}', Boundingbox: {label_bbox}, H:{line_height}")

    # Initialize signature area coordinates (will be populated later)
    # 初始化簽名區域坐標(後續會填充)
    sig_xmin, sig_ymin, sig_xmax, sig_ymax = -1, -1, -1, -1
    scene = "unknown" # Tracks which inference rule was applied
    horizontal_margin = _HORIZONTAL_MARGIN 

    # --- Calculate Dynamic Default Width (Helper for Horizontal Fallback) ---
    def calculate_dynamic_default_width(start_x: int, direction: str) -> int:
        # Calculate available space based on direction
        if direction == 'right':
             # For right direction: space from start_x to right edge of page minus margin
             available_space = page_width - start_x - horizontal_margin
        else: # direction == 'left'
             # For left direction: space from left edge to start_x minus margin
             available_space = start_x - horizontal_margin

        # Calculate initial width based on line height
        dynamic_w = line_height * default_width_factor_of_height
        # Cap width by absolute max and relative max (to avoid excessive width)
        # 限制寬度以避免空間太大
        capped_w = min(dynamic_w, max_absolute_default_width, max(0, available_space) * max_relative_default_width_factor)
        
        # Ensure width is at least the minimum required signature width
        # 確保寬度不小於最小簽名寬度
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

    # --- Scene Detection: Identify which layout pattern applies ---
    is_below_case = False  # Flag for signature area below the label
    label_stripped = label_text.strip()

    # 1. Check for "Below" Scenario
    # Condition: Label ends with a keyword AND (it's short OR no close token to the right)
    # 蠻多格子類型的最後一個字會是蓋章，簽名，簽章等
    # Check if label ends with any signature-related keyword
    label_ends_with_keyword = any(label_stripped.endswith(kw) for kw in BELOW_LABEL_KEYWORDS)
    if label_ends_with_keyword:
        # Check if there's NO token immediately to the right
        # If no token nearby, likely a signature box below rather than inline
        # 檢查旁邊是否有文字，沒有文字則可能是下方簽名區
        
        # Use a shorter distance for this specific check
        short_check_dist = max(50, int(line_height * 1.5))
        nearest_right_short_dist_check = find_nearest_token_on_line(
            label_bbox, page_tokens, direction='right',
            max_horizontal_dist=short_check_dist,
            y_tolerance_factor=0.6, 
            ignore_chars=":： "        )
        
        # If no nearby token found, assume signature area is below the label
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

    # --- Inference Logic based on Scene Type ---
    # Handle case where signature area is below the label
    if is_below_case:
        # --- Calculate Vertical Position (Y coordinates) ---
        sig_ymin = ly_max + vertical_margin_below  # Start signature area below label with margin
        # Estimate height based on label height multiplied by the factor
        estimated_height = int(line_height * signature_area_height_factor_below)
        # Ensure minimum sensible height for the signature area
        # 確保簽名區域有足夠的高度
        estimated_height = max(estimated_height, _MIN_INFERRED_HEIGHT * 1.5) # Slightly larger min for below case
        sig_ymax = sig_ymin + estimated_height
        if enable_debug_prints: st.info(f"  '{scene}' Vertical: y_min={sig_ymin}, y_max={sig_ymax} (Est. H: {estimated_height})")
        # --- Calculate Horizontal Position (X coordinates) ---
        sig_xmin = lx_min  # Initially align with label's left edge
        sig_xmax = lx_max  # Initially align with label's right edge
        # Ensure minimum width based on either absolute minimum or relative to label height
        # 確保簽名區域有足夠的寬度，根據標籤高度或絕對最小值
        min_req_width = max(min_sig_width, int(line_height * min_width_factor_below))
        current_width = sig_xmax - sig_xmin

        # If current width is too narrow, expand it to meet minimum requirements
        if current_width < min_req_width:
            needed_expansion = min_req_width - current_width
            # Expand from center outward, respecting page boundaries
            # 向兩邊擴展，但要考慮頁面邊界
            expand_left = needed_expansion // 2  # Try to expand half to the left
            # Calculate new left edge and ensure it's within page boundary
            potential_xmin = sig_xmin - expand_left
            actual_xmin = max(0, potential_xmin)  # Don't go beyond left page edge
            actual_left_expansion = sig_xmin - actual_xmin # How much we actually expanded left

            # Calculate right expansion needed based on how much left expansion was actually possible
            expand_right = needed_expansion - actual_left_expansion  # Remainder goes to the right
            potential_xmax = sig_xmax + expand_right
            actual_xmax = min(page_width, potential_xmax)  # Don't go beyond right page edge

            sig_xmin = actual_xmin
            sig_xmax = actual_xmax

            # Re-check width after boundary adjustments
            final_width = sig_xmax - sig_xmin
            if final_width < min_sig_width * 0.9: # Allow slight tolerance
                 if enable_debug_prints: st.info(f":warning: Warning: '{scene}' Expanded Width ({final_width}) still narrow than  ~{min_sig_width}。")
                #  logging.warning(f"Warning: '{scene}' Expanded Width ({final_width}) still narrow than  ~{min_sig_width}。")
            scene += "_expand_width"
            if enable_debug_prints: st.info(f"  '{scene}' Width expanded from {current_width} to -> {final_width} (Target: {min_req_width}) => xmin={sig_xmin}, xmax={sig_xmax}")
            # logging.info(f"  '{scene}' Width expanded from {current_width} to -> {final_width} (Target: {min_req_width}) => xmin={sig_xmin}, xmax={sig_xmax}")
        else:
             if enable_debug_prints: st.info(f"  '{scene}' Horizontal: x_min={sig_xmin}, x_max={sig_xmax} (Width {current_width} >= {min_req_width})")
            #  logging.info(f"  '{scene}' Horizontal: x_min={sig_xmin}, x_max={sig_xmax} (Width {current_width} >= {min_req_width})")
    
    # Handle case where the label contains both a colon and parentheses
    # Common pattern: "Label: (signature)"
    elif ((':' in label_text or '：' in label_text) and '(' in label_text and ')' in label_text):
        scene = "right_colon_with_paren"
        if enable_debug_prints: st.info(f"scenario: {scene}")

        # --- Calculate label dimensions ---
        label_width = lx_max - lx_min  # Width of the label
        label_height = ly_max - ly_min  # Height of the label
        
        # --- Calculate signature box position (centered on the label) ---
        # Find the center point coordinates of the label
        # 計算標籤的中心點
        center_x = (lx_min + lx_max) / 2
        center_y = (ly_min + ly_max) / 2
        
        # Calculate signature width as percentage of label width, with minimum threshold
        # 計算簽名區域寬度，使用標籤寬度的百分比，但不小於最小寬度
        sig_width = max(label_width * paren_sig_width_percent, paren_sig_min_width)
        
        # Calculate signature height based on label height, ensuring minimum threshold
        # 計算簽名區域高度，不小於最小高度
        sig_height = max(label_height * signature_area_height_factor, _MIN_INFERRED_HEIGHT)
        
        # Calculate signature box coordinates (centered on label's center point)
        # 計算簽名區域的座標，以標籤中心點為基準
        sig_xmin = int(center_x - sig_width / 2)  # Left edge
        sig_xmax = int(center_x + sig_width / 2)  # Right edge
        sig_ymin = int(center_y - sig_height / 2)  # Top edge
        sig_ymax = int(center_y + sig_height / 2)
        
        if enable_debug_prints:
            st.info(f"  Using centered approach for '{scene}'")
            st.info(f"  Label dimensions: width={label_width}, height={label_height}")
            st.info(f"  Center point: ({center_x}, {center_y})")
            st.info(f"  Signature area: ({sig_xmin}, {sig_ymin}, {sig_xmax}, {sig_ymax})")
            st.info(f"  Signature dimensions: width={sig_width}, height={sig_height}")

    # 2. Check for Horizontal Scenarios 
    # Scenario of right colon 簽名:______, 簽章:     
    elif label_stripped.endswith(':') or label_stripped.endswith('：'):
        scene = "right_colon"
        if enable_debug_prints: 
          st.info(f"scenario: {scene}")
        
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
            y_tolerance_factor=nearest_token_y_factor
        )
        if nearest_right_token:
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
        if enable_debug_prints: st.info(f"Scenario: {scene}")
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
            y_tolerance_factor=nearest_token_y_factor        
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
            y_tolerance_factor=nearest_token_y_factor
        )
        if nearest_right_token:
             nr_bbox = nearest_right_token["bbox"]
             sig_xmax = max(sig_xmin + min_sig_width // 2, nr_bbox[0] - horizontal_margin)
             if enable_debug_prints: st.info(f"  '{scene}' Found right token: '{nearest_right_token['text']}' => sig_xmax={sig_xmax}")
            #  logging.info(f"  '{scene}' Found right token: '{nearest_right_token['text']}' => sig_xmax={sig_xmax}")
        else:
             dynamic_width = calculate_dynamic_default_width(sig_xmin, 'right')
             sig_xmax = min(sig_xmin + dynamic_width, page_width - horizontal_margin)
             if enable_debug_prints: st.info(f"  '{scene}' No right token found, using dynamic width {dynamic_width} => sig_xmax={sig_xmax}")
            #  logging.info(f"  '{scene}' No right token found, using dynamic width {dynamic_width} => sig_xmax={sig_xmax}")

    # --- Final Checks 
    if sig_xmin == -1 or sig_xmax == -1 or sig_ymin == -1 or sig_ymax == -1:
        if enable_debug_prints: 
          st.error(f"Error: Cannot calculate all the bounding box (xmin={sig_xmin}, ymin={sig_ymin}, xmax={sig_xmax}, ymax={sig_ymax})。")
        # logging.error(f"Error: Cannot calculate all the bounding box (xmin={sig_xmin}, ymin={sig_ymin}, xmax={sig_xmax}, ymax={sig_ymax})。")
        return None

    # Final width check - ensure min_sig_width is met after all logic
    calculated_width = sig_xmax - sig_xmin
    if calculated_width < min_sig_width:
         if enable_debug_prints: 
          st.warning(f"Warining: Final width ({calculated_width:.0f}) < {min_sig_width}。Fixing....。")
        #  logging.warning(f"Warining: Final width ({calculated_width:.0f}) < {min_sig_width}。Fixing....。")

         if scene.startswith("left"): # Expand left if it was a left-based rule
             sig_xmin = max(0, sig_xmax - min_sig_width)
         elif scene.startswith("right_colon_with_paren"): # Expand right if it was a right-based rule
             pass 
         else: # Default expand right
             sig_xmax = min(page_width, sig_xmin + min_sig_width)
         # If still too narrow (page too small), it is what it is
         if sig_xmax - sig_xmin < min_sig_width * 0.8:
              if enable_debug_prints: st.info(f"  Width fixed ({sig_xmax - sig_xmin}) not enough")
              # logging.info(f"  Width fixed ({sig_xmax - sig_xmin}) not enough")
         else:
              scene += "_final_min_width_fix"
              if enable_debug_prints: st.info(f"  Width fixed: xmin={sig_xmin}, xmax={sig_xmax}")
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
         if enable_debug_prints: 
          st.error(f"Error: Invalid Bounding box:  {inferred_bbox}")
        #  logging.error(f"Error: Invalid Bounding box:  {inferred_bbox}")
         return None

    if enable_debug_prints: 
      st.info(f"Final BBox ('{scene}'): {inferred_bbox}")
    # logging.info(f"Final BBox ('{scene}'): {inferred_bbox}")
    return inferred_bbox, scene

def process_document_ai(file_path: str) -> Optional[documentai.Document]:
    global client, processor_name # Use global client and processor name
    client = documentai.DocumentProcessorServiceClient()
    if not file_path:
        st.error("Error: No file provided")
        return None

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

        st.info(f"""Handling: {file_path}
        \nMIME type: {mime_type}""")

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

        
        progress_text = "Kdan AI processing in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(50):  # 先顯示一半的進度
            time.sleep(0.02)  # 較短的延遲，讓動畫更快
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        result = documentai_client.process_document(request=request)
        
        for percent_complete in range(50, 100):
            time.sleep(0.01)  # 較短的延遲
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        time.sleep(0.5)
        my_bar.empty()
        st.success("Kdan AI processing completed successfully! 🎉")

        return result.document

    except Exception as e:
        st.error(f"Handling {file_path} Error: {e}")
        return None
        

def send_feedback_to_bigquery(metadata: Dict):
    """Send feedback and metadata to BigQuery."""
    from google.cloud import bigquery
    
    # Initialize BigQuery client
    client = bigquery.Client()
    
    # Specify your BigQuery dataset and table
    project_id = "kdan-it-playground"
    dataset_id = "test_bennett"
    table_id = "streamlit_signature_detection"
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    # Check if table exists, if not create it
    try:
        client.get_table(full_table_id)
    except Exception:
        # Table doesn't exist, create it with schema
        schema = [
            bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("feedback", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("results", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("comment", "STRING", mode="NULLABLE")
        ]
        
        # Create the table
        table = bigquery.Table(full_table_id, schema=schema)
        table = client.create_table(table)  # Make an API request
        st.success(f"Created table {full_table_id}")
    
    # Insert data into BigQuery
    errors = client.insert_rows_json(
        full_table_id, 
        [metadata]
    )
    
    if errors:
        raise Exception(f"Errors inserting rows to BigQuery: {errors}")
    return True

def extract_and_infer_signature_areas(document: documentai.Document) -> Tuple[List[Dict], Dict, Dict]:
    results = []
    doc_ai_dimensions = {}
    page_images = {}  # 儲存所有頁面的圖像

    if not document.pages:
        st.warning("Warning: Document AI has no page details.")
        return results, doc_ai_dimensions, page_images

    page_tokens_map = {}
    # Pre-process all pages to get tokens, dimensions, and images
    for i, page in enumerate(document.pages):
        if not page.dimension or not page.dimension.width or not page.dimension.height or page.dimension.width <= 0 or page.dimension.height <= 0:
            st.warning(f"Warning: Page {i} invalid. Skip")
            continue

        page_width_docai = int(page.dimension.width)
        page_height_docai = int(page.dimension.height)
        doc_ai_dimensions[i] = {'width': page_width_docai, 'height': page_height_docai}

        # Extract tokens for the current page
        page_tokens = get_page_tokens(page, page_width_docai, page_height_docai)
        page_tokens_map[i] = {
            "tokens": page_tokens,
            "width": page_width_docai,
            "height": page_height_docai
        }

        # Decode page image if available
        if hasattr(page, 'image') and page.image.content:
            try:
                # 將二進制圖像數據轉換為 OpenCV 格式
                nparr = np.frombuffer(page.image.content, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    page_images[i] = img_cv
                else:
                    st.warning(f"Warning: Failed to decode image for page {i}")
            except Exception as e:
                st.error(f"Error decoding image for page {i}: {e}")

    if not document.entities:
        st.warning("Warning: Document AI has not label entities.")

    target_entity_type = "signature_field"
    if enable_debug_prints:
        st.info(f"From {len(document.entities)} Entities searching '{target_entity_type}'...")

    processed_entity_count = 0

    for entity in document.entities:
        if entity.type == target_entity_type:
            if not entity.page_anchor or not entity.page_anchor.page_refs:
                st.warning(f"Warning: Entity '{entity.mention_text}' has no page_anchor or page_refs")
                continue

            page_ref = entity.page_anchor.page_refs[0]
            page_num = page_ref.page

            if page_num not in page_tokens_map:
                st.warning(f"Warning: The page {page_num} corresponding to entity '{entity.mention_text}' has not been processed.")
                continue

            page_info = page_tokens_map[page_num]
            page_width_docai = page_info["width"]
            page_height_docai = page_info["height"]
            page_tokens = page_info["tokens"]

            if not page_ref.bounding_poly or not page_ref.bounding_poly.normalized_vertices:
                st.warning(f"Warning: The page_ref for entity '{entity.mention_text}' (Page {page_num}) is missing bounding_poly.")
                continue

            label_bbox_normalized = page_ref.bounding_poly.normalized_vertices
            label_bbox_pixel_docai = get_pixel_bbox(label_bbox_normalized, page_width_docai, page_height_docai)

            # 動態計算參數
            min_sig_width_default = calculate_dynamic_min_sig_width(page_tokens, page_width_docai)
            nearest_token_y_factor = calculate_dynamic_y_tolerance(page_tokens)
            nearest_token_max_dist = calculate_dynamic_max_horizontal_dist(page_tokens, page_width_docai)
            signature_area_height_factor = calculate_dynamic_height_factor(label_bbox_pixel_docai, page_tokens)

            inference_result = infer_signature_area_bbox(
                label_entity=entity,
                page_tokens=page_tokens,
                page_width=page_width_docai,
                page_height=page_height_docai,
                min_sig_width=min_sig_width_default,
                signature_area_height_factor=signature_area_height_factor,
                nearest_token_max_dist=nearest_token_max_dist,
                nearest_token_y_factor=nearest_token_y_factor
            )

            inferred_area_bbox_pixel_docai = None
            inferred_rule = "failed"
            inferred_area_bbox_normalized = None

            if inference_result:
                inferred_area_bbox_pixel_docai, inferred_rule = inference_result
                processed_entity_count += 1

                sig_xmin, sig_ymin, sig_xmax, sig_ymax = inferred_area_bbox_pixel_docai
                norm_xmin = sig_xmin / page_width_docai
                norm_ymin = sig_ymin / page_height_docai
                norm_xmax = sig_xmax / page_width_docai
                norm_ymax = sig_ymax / page_height_docai

                inferred_area_bbox_normalized = [
                    documentai.NormalizedVertex(x=norm_xmin, y=norm_ymin),
                    documentai.NormalizedVertex(x=norm_xmax, y=norm_ymin),
                    documentai.NormalizedVertex(x=norm_xmax, y=norm_ymax),
                    documentai.NormalizedVertex(x=norm_xmin, y=norm_ymax)
                ]

            label_bbox_pixel_docai_for_storage = get_pixel_bbox(label_bbox_normalized, page_width_docai, page_height_docai)

            # Append result
            result_dict = {
                "entity_type": target_entity_type,
                "label_text": entity.mention_text,
                "label_bbox_normalized": label_bbox_normalized,
                "label_bbox_pixel_docai": label_bbox_pixel_docai_for_storage,
                "inferred_area_bbox_pixel_docai": inferred_area_bbox_pixel_docai,
                "inferred_area_bbox_normalized": inferred_area_bbox_normalized,
                "inferred_rule": inferred_rule,
                "confidence": entity.confidence,
                "page": page_num,
                "min_sig_width": min_sig_width_default
            }
            results.append(result_dict)

    if enable_debug_prints:
        st.info(f"Found and attempting to process {len([r for r in results if r['entity_type'] == target_entity_type])} '{target_entity_type}' entities.")
        st.info(f"Successfully inferred {processed_entity_count} signature regions.")
    return results, doc_ai_dimensions, page_images

# Initialize session state variables if they don't exist
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'doc_dimensions' not in st.session_state:
    st.session_state.doc_dimensions = None
if 'current_processed_results' not in st.session_state:
    st.session_state.current_processed_results = None
if 'current_file_path' not in st.session_state:
    st.session_state.current_file_path = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_value' not in st.session_state:
    st.session_state.feedback_value = None
if 'feedback_comment' not in st.session_state:
    st.session_state.feedback_comment = ""

# Callback function for handling feedback
def handle_document_feedback(feedback, comment, doc_id, processed_results,page_images):
    """Function to handle document-level feedback submission"""
    feedback_key = f"feedback_doc_{doc_id}"
    
    if feedback is not None:
        # Prepare metadata for BigQuery
        current_time = datetime.now(pytz.timezone('Asia/Taipei')).isoformat()
        
        metadata = {
            "document_id": doc_id,
            "feedback": feedback,
            "comment": comment,
            "timestamp": current_time,
            "results": str(processed_results)  ,
            "page_images": str(page_images)
        }
        
        # Store in session state
        st.session_state.feedback_data[feedback_key] = metadata
        
        # Send metadata to BigQuery
        try:
            send_feedback_to_bigquery(metadata)
            st.toast("Feedback successfully recorded!", icon="✅")
            return True
        except Exception as e:
            st.toast(f"Feedback recording failed: {str(e)}", icon=":x:")
            return False
    return None

def process_feedback():
    """處理反饋提交的回調函數"""
    # 從session state中獲取需要的值
    feedback = st.session_state.feedback_value
    comment = st.session_state.feedback_comment
    
    # 檢查是否有保存的處理結果
    if st.session_state.current_processed_results is not None and st.session_state.current_file_path is not None:
        doc_id = os.path.basename(st.session_state.current_file_path)
        
        # 處理反饋
        if feedback is not None:
            success = handle_document_feedback(feedback, comment, doc_id, st.session_state.current_processed_results)
            if success:
                # 設置提交標誌
                st.session_state.feedback_submitted = True

# Function to handle starting the processing
def start_processing():
    # No need to set any flags, the button click will trigger rerun
    pass

# --- Main Processing Logic ---
if st.session_state.uploaded_file:
    st.text(f"已選擇文件：{st.session_state.uploaded_file.name}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("開始"):
            # Process the file when button is clicked
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
                    doc_id = os.path.basename(file_path)
                    bucket = storage_client.bucket(BUCKET_NAME)
                    now = datetime.now()
                    upload_date = now.strftime("%Y%m%d")
                    timestamp = now.strftime("%H%M%S")
                    mime_type, _ = mimetypes.guess_type(st.session_state.uploaded_file.name)
                    file_type = "pdf" if mime_type == "application/pdf" else "image"
                    original_filename = st.session_state.uploaded_file.name
                    gcs_blob_name = f"streamlit_dataset/{upload_date}/{file_type}/{timestamp}_{doc_id}"
                    blob = bucket.blob(gcs_blob_name)
                    blob.upload_from_filename(file_path)
                    st.session_state.gcs_path = f"gs://{BUCKET_NAME}/{gcs_blob_name}"
                    st.success(f"文件已上傳至 GCS: {st.session_state.gcs_path}")
                except Exception as e:
                    st.error(f"無法上傳文件至 GCS: {e}")

            # --- Adjusted Visualization for Streamlit ---
            def visualize_results(processed_results: List[Dict], doc_ai_dimensions: Dict, page_images: Dict):
                """Visualize processed_results, recalculating coordinates based on the actual image dimensions for drawing."""
                if not doc_ai_dimensions:
                    st.error("No pages to visualize (doc_ai_dimensions is empty).")
                    return

                for page_num in sorted(doc_ai_dimensions.keys()):
                    page_results = [r for r in processed_results if r.get("page") == page_num]
                    if page_num not in page_images or page_images[page_num] is None:
                        st.warning(f"Warning: Page {page_num} not found in page_images.")
                        continue
                    
                    img_cv = page_images[page_num]
                    if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
                        if len(img_cv.shape) == 2:  # Grayscale -> BGR
                            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                        elif img_cv.shape[2] == 4:  # RGBA -> BGR
                            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
                        else:
                            st.error(f"Error: Unable to convert image for page {page_num} to BGR format, skipping.")
                            continue
                    
                    vis_height, vis_width, _ = img_cv.shape
                    st.info(f"--- Visualizing page {page_num + 1} ---")
                    if enable_debug_prints:
                        st.info(f"  Image dimensions: {vis_width}x{vis_height}")

                    img_to_show = img_cv.copy()
                    if page_results:
                        for result_idx, result in enumerate(page_results):
                            if enable_debug_prints:
                                st.info(f"  Processing result {result_idx+1}/{len(page_results)}: Label='{result.get('label_text', 'N/A')}'")
                            entity_type = result.get("entity_type", "unknown_label")
                            inferred_rule = result.get("inferred_rule", "N/A")
                            label_normalized_vertices = result.get("label_bbox_normalized")
                            inferred_normalized_vertices = result.get("inferred_area_bbox_normalized")
                            label_bbox_vis = None
                            if label_normalized_vertices:
                                x_coords = [v.x for v in label_normalized_vertices if v.x is not None]
                                y_coords = [v.y for v in label_normalized_vertices if v.y is not None]
                                if x_coords and y_coords:
                                    lx_vis = int(min(x_coords) * vis_width)
                                    ly_vis = int(min(y_coords) * vis_height)
                                    lx2_vis = int(max(x_coords) * vis_width)
                                    ly2_vis = int(max(y_coords) * vis_height)
                                    label_bbox_vis = (lx_vis, ly_vis, lx2_vis, ly2_vis)
                            
                            if label_bbox_vis and lx_vis < lx2_vis and ly_vis < ly2_vis:
                                try:
                                    cv2.rectangle(img_to_show, (lx_vis, ly_vis), (lx2_vis, ly2_vis), (0, 0, 255), 2)  # 紅色
                                    text_y = ly_vis - 10 if ly_vis > 15 else ly2_vis + 20
                                    label_text_simple = f"{entity_type}"
                                    cv2.putText(img_to_show, label_text_simple, (lx_vis, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                except Exception as draw_err:
                                    st.error(f"Error: Failed to draw Label '{entity_type}': {draw_err}, BBox={label_bbox_vis}")
                            inferred_bbox_vis = None
                            if inferred_normalized_vertices:
                                x_coords = [v.x for v in inferred_normalized_vertices if v.x is not None]
                                y_coords = [v.y for v in inferred_normalized_vertices if v.y is not None]
                                if x_coords and y_coords:
                                    ix_vis = int(min(x_coords) * vis_width)
                                    iy_vis = int(min(y_coords) * vis_height)
                                    ix2_vis = int(max(x_coords) * vis_width)
                                    iy2_vis = int(max(y_coords) * vis_height)
                                    inferred_bbox_vis = (ix_vis, iy_vis, ix2_vis, iy2_vis)
                            if inferred_bbox_vis and ix_vis < ix2_vis and iy_vis < iy2_vis:
                                try:
                                    cv2.rectangle(img_to_show, (ix_vis, iy_vis), (ix2_vis, iy2_vis), (0, 255, 0), 2)  # 綠色
                                    text_y = iy_vis - 10 if iy_vis > 15 else iy2_vis + 20
                                    rule_str = str(inferred_rule).replace("_", " ")
                                    inferred_text_simple = f"{rule_str}"
                                    if abs(iy_vis - ly_vis) < 30 and ix_vis < lx2_vis + 50: 
                                        text_y = iy2_vis + 20
                                    cv2.putText(img_to_show, inferred_text_simple, (ix_vis, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
                                except Exception as draw_err:
                                    st.error(f"Error: Failed to draw Inferred box/text: {draw_err}, BBox={inferred_bbox_vis}")
                            elif label_bbox_vis:
                                try:
                                    lx2_vis, ly_vis, _, _ = label_bbox_vis
                                    cv2.putText(img_to_show, 'Area Infer Failed', (lx2_vis + 5, ly_vis + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                                    st.info(f"Marked inference failure (Rule: {inferred_rule})")
                                except Exception as draw_err:
                                    st.error(f"Error: Failed to draw Label '{entity_type}': {draw_err}, BBox={label_bbox_vis}")
                    else:
                        st.warning(f"  Page {page_num + 1} has no signature fields, displaying original image.")
                    try:
                        img_to_show_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
                        st.image(img_to_show_rgb, caption=f"Page {page_num + 1} - Signature Area Inference (DocAI Label: Red, Post-Processing: Green)")
                    except Exception as plot_err:
                        st.error(f"Error: Failed to display page {page_num} with Streamlit: {plot_err}")
            document = process_document_ai(file_path)
                
            if document:
                processed_results, doc_ai_dimensions,page_images = extract_and_infer_signature_areas(document)
                # Store results in session state
                st.session_state.processed_results = processed_results
                st.session_state.doc_dimensions = doc_ai_dimensions
                st.session_state.page_images = page_images
                st.session_state.current_processed_results = processed_results
                st.session_state.current_file_path = file_path
            else:
                st.error("Failed to process document with Kdan AI")
                st.stop()
                    
            # Use the results
            processed_results = st.session_state.processed_results
            doc_ai_dimensions = st.session_state.doc_dimensions
            page_images = st.session_state.page_images
                
            st.subheader(":the_horns: Kdan AI result")
            visualize_results(processed_results, doc_ai_dimensions,page_images)
            if processed_results:
                st.success(":the_horns: **Signature area detected!** :the_horns:")
                st.text("Bounding Box:")
                st.write(processed_results)
            else:
                st.warning("rolling_on_the_floor_laughing: No signature detected or processing error. rolling_on_the_floor_laughing:")

            # --- Feedback Section ---
            st.subheader(":thinking_face: **Feedback** :thinking_face: ")
            st.caption("請大力鞭策！")
            doc_id = os.path.basename(st.session_state.current_file_path)
            feedback_key = f"feedback_doc_{doc_id}"
            
            if feedback_key in st.session_state.feedback_data:
                previous_feedback = st.session_state.feedback_data[feedback_key]['feedback']
                st.info(f"Previous Feedback: {previous_feedback}")
            
            # Create a form for unified document feedback
            with st.form(key="feedback_form_doc"):
                st.caption("**Your feedback is important to us!!! :heart:**")
                feedback = st.feedback("thumbs", key="feedback_value")
                comment = st.text_area("Comment (Optional)", key="feedback_comment")
                submitted = st.form_submit_button("Submit Feedback", on_click=process_feedback)
            
            if st.session_state.feedback_submitted:
                st.success("Feedback submitted and logged!")
                st.session_state.feedback_submitted = False
    with col2:
        if st.button("Clear"):
            # Reset all relevant session state variables
            st.session_state.uploaded_file = None
            st.session_state.processed_results = None
            st.session_state.doc_dimensions = None
            st.session_state.page_images = None
            st.session_state.gcs_path = None
            st.session_state.uploader_key = f"doc_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.file_source = "Upload File" # Reset file source to default
            st.session_state.process_started = False
            st.session_state.processing_complete = False
            st.session_state.feedback_data = {}
            # Clear all keys that start with 'processed_file_' or 'feedback_page_'
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('processed_file_') or k.startswith('feedback_page_')]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
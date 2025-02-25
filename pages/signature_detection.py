import streamlit as st
import os
import json
from google.cloud import documentai_v1 as documentai
from pdf2image import convert_from_path
import cv2
import numpy as np
import mimetypes
import tempfile
from dotenv import load_dotenv

load_dotenv()
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND")

# 判斷是本地還是正式環境
if os.path.isfile(".env") or os.getenv("ENV") == "dev":  
    if GOOGLE_CREDENTIALS and os.path.isfile(GOOGLE_CREDENTIALS):
        with open(GOOGLE_CREDENTIALS, "r") as f:
            DEFAULT_SA_KEY = json.load(f)
    else:
        st.error("本地環境：請在 .env 中設定 GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND 為有效 JSON 檔案路徑")
        DEFAULT_SA_KEY = None
else:  # 正式環境（如 Streamlit Cloud）
    if GOOGLE_CREDENTIALS:
        try:
            DEFAULT_SA_KEY = json.loads(GOOGLE_CREDENTIALS)  # 解析 JSON 字串
        except json.JSONDecodeError:
            st.error("正式環境：GOOGLE_CREDENTIALS_KDAN_IT_PLAYGROUND 必須是有效的 JSON 字串")
            DEFAULT_SA_KEY = None
    else:
        DEFAULT_SA_KEY = None


if DEFAULT_SA_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_file:
        json.dump(DEFAULT_SA_KEY, tmp_file)
        tmp_file.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name  # 使用正確的環境變數名稱
else:
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        st.error("未找到有效的 Google Cloud 憑證，請檢查環境設定")

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

# 初始化 Document AI 客戶端
try:
    client = documentai.DocumentProcessorServiceClient()
except Exception as e:
    st.error(f"無法初始化 Document AI 客戶端，請檢查憑證：{e}")
    st.stop()

processor_name = "projects/962438265955/locations/us/processors/f69f1e73163aad4a"

# 後續程式碼保持不變
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'boxes' not in st.session_state:
    st.session_state.boxes = None

st.subheader("Upload Target File")
st.write("Only support format with PDF (within 15 pages), JPG, JPEG, PNG")
uploaded_file = st.file_uploader("SELECT FILE", type=["pdf", "jpg", "jpeg", "png"], key="doc")

if st.button("Clear"):
    st.session_state.uploaded_file = None
    st.session_state.boxes = None
    st.rerun()

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(st.session_state.uploaded_file.read())
        file_path = tmp_file.name

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

            result = client.process_document(request=request)
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
                cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"Conf: {confidence:.2f}"
                cv2.putText(img_cv, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
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
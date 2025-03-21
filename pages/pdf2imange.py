import streamlit as st
from pdf2image import convert_from_bytes
import io
from PIL import Image
import os

def convert_pdf_to_images(pdf_file):
    # 將 PDF 轉換為圖片列表
    images = convert_from_bytes(pdf_file.read())
    return images

def main():
    st.title("PDF 轉圖片服務")
    st.write("上傳您的 PDF 文件，將其轉換為圖片並選擇下載！")

    # 文件上傳
    uploaded_file = st.file_uploader("選擇一個 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        # 轉換 PDF 到圖片
        with st.spinner("正在轉換 PDF 到圖片..."):
            images = convert_pdf_to_images(uploaded_file)
        
        # 顯示圖片數量
        st.success(f"已成功轉換！共 {len(images)} 頁")
        
        # 讓用戶選擇要下載的頁面
        selected_pages = st.multiselect(
            "選擇要下載的頁面",
            options=[f"第 {i+1} 頁" for i in range(len(images))],
            default=[f"第 {i+1} 頁" for i in range(len(images))]
        )

        # 原始檔案名稱作為預設值
        default_base_filename = uploaded_file.name.split('.')[0]

        # 顯示所選頁面的圖片
        if selected_pages:
            st.subheader("圖片預覽")
            for page in selected_pages:
                page_num = int(page.split()[1]) - 1
                img = images[page_num]
                
                # 顯示圖片
                st.image(img, caption=page, use_container_width=True)
                
                # 為每個圖片創建獨立的檔案名稱輸入框
                col1, col2 = st.columns([3, 1])  # 分成兩欄，輸入框占3份，下載按鈕占1份
                with col1:
                    default_filename = f"{default_base_filename}_page_{page_num + 1}"
                    filename = st.text_input(
                        f"檔案名稱（{page}）",
                        value=default_filename,
                        key=f"filename_{page_num}"
                    )
                
                # 將圖片轉換為字節
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # 下載按鈕
                with col2:
                    st.download_button(
                        label=f"下載",
                        data=img_byte_arr,
                        file_name=f"{filename}.png",
                        mime="image/png",
                        key=f"download_{page_num}"
                    )

if __name__ == "__main__":
    main()
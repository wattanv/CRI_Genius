# =================================================================
# 1. Import ไลบรารีที่จำเป็นทั้งหมด
# =================================================================
import streamlit as st
from datetime import datetime
from PIL import Image
from collections import Counter
import numpy as np
import cv2
import io
import os
from roboflow import Roboflow
import time
import shutil
import pandas as pd
import plotly.express as px
import random

# =================================================================
# 2. ตั้งค่าหน้าเว็บ, UI Styles, และโหลดโมเดล
# =================================================================
st.set_page_config(page_title="CRI Genius", layout="wide")

# --- โค้ดสำหรับ UI Styles และ Header ---
st.markdown(
    """
    <div style="
        background: linear-gradient(to right, #6FC3FF, #2176FF);
        border-radius: 12px;
        padding: 10px 0;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    ">
        CRI Genius
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
.inline-label {
    display: flex; align-items: center; gap: 8px; font-weight: bold;
    font-size: 18px; color: black; margin-bottom: 5px;
}
.inline-label img { width: 24px; height: 24px; }
.stTextInput input, .stDateInput input, .stSelectbox > div > div {
    background-color: #e6f2ff !important; border-radius: 8px !important; color: black !important;
}
.stTextInput input:disabled {
    background-color: #d0e0f0 !important; font-weight: bold; text-align: center; color: black !important;
}
div.stButton > button {
    background: linear-gradient(to right, #6FC3FF, #2176FF); color: black;
    font-weight: bold; border-radius: 25px; padding: 10px 40px; font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันสำหรับโหลดโมเดล AI (ใช้ cache เพื่อความเร็ว) ---
@st.cache_resource
def load_models():
    """โหลดโมเดลจาก Roboflow และคืนค่าเป็น tuple"""
    try:
        # **สำคัญมาก:** กรุณาใส่ API KEY ที่ถูกต้องของคุณ
        DETECTOR_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        CLASSIFIER_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        
        detector_model = Roboflow(api_key=DETECTOR_API_KEY).workspace("wattanathornch").project("crystal_quality_detection").version(9).model
        classifier_model = Roboflow(api_key=CLASSIFIER_API_KEY).workspace("wattanathornch").project("crystal_quality").version(1).model
        return detector_model, classifier_model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None, None

# --- ฟังก์ชันสำหรับบันทึกข้อมูล ---
def save_results_to_csv(data, filename="analysis_history.csv"):
    df = pd.DataFrame([data])
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

# --- ฟังก์ชัน Helper: ปรับขนาดภาพพร้อม Padding ---
def resize_with_padding(image, target_size=(640, 640), color=(128, 128, 128)):
    original_w, original_h = image.size
    target_w, target_h = target_size
    ratio = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)
    
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, color)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    new_image.paste(resized_image, (offset_x, offset_y))
    return new_image, ratio, (offset_x, offset_y)

# สั่งให้โหลดโมเดลและเตรียม Session State
detector_model, classifier_model = load_models()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# =================================================================
# 3. ส่วน UI และ Logic การทำงานหลัก
# =================================================================
col_main_left, col_main_right = st.columns([1, 1.2])

# --- UI ฝั่งซ้าย (Input) ---
with col_main_left:
    st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/calendar--v1.png"/><span>วันที่วิเคราะห์</span></div>""", unsafe_allow_html=True)
    st.date_input("", datetime.today(), key="date_input", label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/edit--v1.png"/><span>Stike</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="stike", label_visibility="collapsed")
    with col2:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/bookmark-ribbon--v1.png"/><span>class</span></div>""", unsafe_allow_html=True)
        st.selectbox("", ["SR", "R", "W100", "W150"], key="class_selection", label_visibility="collapsed")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/user.png"/><span>ผู้วิเคราะห์</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="analyst", label_visibility="collapsed", help="พิมพ์ 'test' เพื่อรันโหมดทดสอบ")
    with col4:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/worker-male.png"/><span>ช่างเคี่ยว</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="operator", label_visibility="collapsed")

    uploaded_file = st.file_uploader("### 📷 เลือกรูปที่นี่", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.markdown("### ⚙️ ตั้งค่าการวิเคราะห์")
    confidence_threshold = st.slider(
        "เกณฑ์ความมั่นใจ Classifier (%)", 0, 100, 40, 5,
        help="AI จะนับเฉพาะผลึกที่จำแนกได้ด้วยความมั่นใจมากกว่าค่าที่ตั้งไว้"
    )

    process_button = st.button("ประมวลผล")

# --- Logic การประมวลผล ---
if process_button:
    st.session_state.analysis_results = None 
    if uploaded_file is None:
        st.warning("⚠️ กรุณาอัปโหลดรูปภาพก่อน")
    elif not detector_model or not classifier_model:
        st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน")
    elif st.session_state.get("analyst") == "test":
        st.info("โหมดทดสอบ: กำลังใช้ข้อมูลปลอม")
        # ... (โค้ดโหมดทดสอบ) ...
        st.rerun()
    else:
        try:
            with st.spinner("🧠 กำลังประมวลผลด้วย AI..."):
                image_pil_original = Image.open(uploaded_file).convert("RGB")
                
                # 1. ปรับขนาดภาพสำหรับ Detector
                main_image_pil_resized, ratio, offset = resize_with_padding(image_pil_original, target_size=(640, 640))
                temp_path_detector = "temp_detector.jpg"
                main_image_pil_resized.save(temp_path_detector)
                detections = detector_model.predict(temp_path_detector, confidence=10, overlap=50).json().get('predictions', [])
                os.remove(temp_path_detector)
                
                # 2. Classification
                full_results = []
                temp_crop_dir = "temp_crops"; os.makedirs(temp_crop_dir, exist_ok=True)
                offset_x, offset_y = offset
                progress_bar = st.progress(0, text="กำลังจำแนกประเภท...")

                for i, detection in enumerate(detections):
                    try:
                        box_center_x,y,w,h = detection['x'],detection['y'],detection['width'],detection['height']
                        orig_center_x = (box_center_x - offset_x) / ratio
                        orig_center_y = (y - offset_y) / ratio
                        orig_w, orig_h = w / ratio, h / ratio
                        x1,y1,x2,y2 = int(orig_center_x-orig_w/2),int(orig_center_y-orig_h/2),int(orig_center_x+orig_w/2),int(orig_center_y+orig_h/2)
                        cropped_crystal_img = image_pil_original.crop((x1, y1, x2, y2))

                        resized_crop, _, _ = resize_with_padding(cropped_crystal_img, target_size=(224, 224))
                        temp_predict_path = os.path.join(temp_crop_dir, f"predict_{i}.jpg")
                        resized_crop.save(temp_predict_path)

                        final_class, final_confidence = None, 0.0
                        try:
                            raw_result = classifier_model.predict(temp_predict_path).json()
                            if 'predictions' in raw_result and raw_result['predictions'] and 'predictions' in raw_result['predictions'][0]:
                                inner_predictions = raw_result['predictions'][0]['predictions']
                                if inner_predictions:
                                    top_pred = inner_predictions[0]
                                    api_class, api_conf = top_pred.get('class'), top_pred.get('confidence', 0.0)
                                    if api_class and (api_conf * 100) >= confidence_threshold:
                                        final_class, final_confidence = api_class, api_conf
                        except Exception: pass
                        full_results.append({"class": final_class, "confidence": final_confidence})
                    except Exception: continue
                    progress_bar.progress((i + 1) / len(detections))
                
                progress_bar.empty()
                if os.path.exists(temp_crop_dir): shutil.rmtree(temp_crop_dir)

                # 3. คำนวณสรุปผล
                classified_classes = [res['class'] for res in full_results if res.get('class')]
                total_grains = len(classified_classes)
                grade_counts = Counter(classified_classes)
                N3,N2,N1,N0 = grade_counts.get('class 3',0),grade_counts.get('class 2',0),grade_counts.get('class 1',0),grade_counts.get('class 0',0)
                
                cri_score = 0.0
                if total_grains > 0:
                     numerator, denominator = (3*N3 + 2*N2 + 1*N1), 3 * total_grains
                     cri_score = (numerator / denominator) * 100

                st.session_state.analysis_results = {"N3":N3,"N2":N2,"N1":N1,"N0":N0,"total_grains":total_grains,"total_detected":len(detections),"cri":cri_score,"processed_image":image_pil_original}

            st.success("✅ ประมวลผลสำเร็จ!")
        except Exception as e:
            st.error(f"😭 เกิดข้อผิดพลาด: {e}")
        st.rerun()
        
with col_main_right:
    results = st.session_state.analysis_results

    if results:
        st.image(results["processed_image"], caption="ภาพที่วิเคราะห์", use_container_width=True)
    elif uploaded_file:
        st.image(uploaded_file, caption="ภาพที่เลือก", use_container_width=True)
    else:
        placeholder_image = np.full((400, 600, 3), 240, dtype=np.uint8)
        cv2.putText(placeholder_image, "Upload an image to start", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        st.image(placeholder_image, caption="ผลการวิเคราะห์จะแสดงที่นี่", use_container_width=True)

    if results:
        st.info(f"AI ตรวจพบผลึก **{results.get('total_detected',0)}** ชิ้น และจำแนกได้อย่างมั่นใจ **{results.get('total_grains',0)}** ชิ้น")

    st.markdown("### ผลการประเมินคะแนนเม็ดน้ำตาล")

    N3, N2, N1, N0 = (results.get(k,0) for k in ["N3","N2","N1","N0"]) if results else (0,0,0,0)
    total_grains, cri = (results.get(k,0) for k in ["total_grains","cri"]) if results else (0,0.0)
    
    p3, p2, p1, p0 = ((n/total_grains*100) if total_grains > 0 else 0 for n in [N3,N2,N1,N0])
    
    data_to_display = [{"class": 3, "count": N3, "percent": p3}, {"class": 2, "count": N2, "percent": p2},
                       {"class": 1, "count": N1, "percent": p1}, {"class": 0, "count": N0, "percent": p0}]

    for item in data_to_display:
        g, c, m, p, pct = st.columns([1, 1, 0.5, 1, 0.5])
        with g: st.markdown(f"คะแนน {item['class']}")
        with c: st.text_input("", str(item['count']), disabled=True, key=f"d_c_{item['class']}", label_visibility="collapsed")
        with m: st.markdown("เม็ด")
        with p: st.text_input("", f"{item['percent']:.2f}", disabled=True, key=f"d_p_{item['class']}", label_visibility="collapsed")
        with pct: st.markdown("%")

    t1,t2,t3,t4,t5 = st.columns([1,1,0.5,1,0.5])
    with t1: st.markdown("**Total**")
    with t2: st.markdown(f"**{total_grains}**")
    with t3: st.markdown("เม็ด")
    with t4: st.markdown(f"**{p3+p2+p1+p0:.2f}**")
    with t5: st.markdown("%")
    
    st.markdown("### **%CRI**")
    st.text_input("", f"{cri:.2f} %", disabled=True, key="d_cri", label_visibility="collapsed")

    if st.button("บันทึกข้อมูล", key="save_button"):
        if results:
            try:
                record_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **{k: st.session_state.get(k) for k in ["date_input","stike","class_selection","analyst","operator"]}, **{k: results.get(k,0) for k in ["total_detected","total_grains","N3","N2","N1","N0"]}, "cri_score": f"{results.get('cri',0.0):.2f}"}
                save_results_to_csv(record_data)
                st.success(f"✅ บันทึกข้อมูลเรียบร้อยแล้ว!")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการบันทึก: {e}")
        else:
            st.error("❌ ไม่มีข้อมูลให้บันทึก")

    CSV_FILENAME = "analysis_history.csv"
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "rb") as f:
            st.download_button(label="📥 ดาวน์โหลดประวัติ (CSV)", data=f, file_name=CSV_FILENAME, mime="text/csv", key="download_csv_button")

# =================================================================
# 4. ส่วนแสดงกราฟ
# =================================================================
with col_main_left:
    if st.session_state.analysis_results:
        results_for_graph = st.session_state.analysis_results
        total_grains_for_graph = results_for_graph.get("total_grains", 0)

        if total_grains_for_graph > 0:
            st.markdown("---")
            st.subheader("กราฟสัดส่วนผลึก (%)")

            n_values = {
                "คะแนน 3": results_for_graph.get("N3", 0),
                "คะแนน 2": results_for_graph.get("N2", 0),
                "คะแนน 1": results_for_graph.get("N1", 0),
                "คะแนน 0": results_for_graph.get("N0", 0)
            }
            
            chart_data = {
                'ประเภทคะแนน': list(n_values.keys()),
                'จำนวน': list(n_values.values())
            }
            df_chart = pd.DataFrame(chart_data)

            df_chart['เปอร์เซ็นต์'] = (df_chart['จำนวน'] / total_grains_for_graph) * 100

            fig_bar = px.bar(
                df_chart[df_chart['จำนวน'] > 0], 
                x='ประเภทคะแนน', 
                y='เปอร์เซ็นต์', 
                color='ประเภทคะแนน',
                text_auto='.2f', 
                color_discrete_map={
                    'คะแนน 3': '#4CAF50', 
                    'คะแนน 2': '#8BC34A',
                    'คะแนน 1': '#FFC107', 
                    'คะแนน 0': '#F44336'
                },
                labels={'เปอร์เซ็นต์': 'สัดส่วน (%)', 'ประเภทคะแนน': ''}
            )

            fig_bar.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig_bar.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None)
            
            st.plotly_chart(fig_bar, use_container_width=True)

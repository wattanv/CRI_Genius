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

@st.cache_resource
def load_models():
    """โหลดโมเดลจาก Roboflow และคืนค่าเป็น tuple"""
    try:
        DETECTOR_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        CLASSIFIER_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        detector_model = Roboflow(api_key=DETECTOR_API_KEY).workspace("wattanathornch").project("crystal_quality_detection").version(9).model
        classifier_model = Roboflow(api_key=CLASSIFIER_API_KEY).workspace("wattanathornch").project("crystal_quality").version(1).model
        return detector_model, classifier_model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None, None

def save_results_to_csv(data, filename="analysis_history.csv"):
    df = pd.DataFrame([data])
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

detector_model, classifier_model = load_models()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# =================================================================
# 3. ส่วน UI และ Logic การทำงานหลัก
# =================================================================
col_main_left, col_main_right = st.columns([1, 1.2])

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
        st.text_input("", key="analyst", label_visibility="collapsed", help="พิมพ์ 'test1' (สุ่ม) หรือ 'test2' (ค่าคงที่) เพื่อรันโหมดทดสอบ")
    with col4:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/worker-male.png"/><span>ช่างเคี่ยว</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="operator", label_visibility="collapsed")

    uploaded_file = st.file_uploader("### 📷 เลือกรูปที่นี่", type=["jpg", "jpeg", "png"])
    
    process_button = st.button("ประมวลผล")

if process_button:
    st.session_state.analysis_results = None 

    if uploaded_file is None:
        st.warning("⚠️ กรุณาอัปโหลดรูปภาพก่อนกดประมวลผล")
    elif not detector_model or not classifier_model:
        st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน ไม่สามารถประมวลผลได้")

    elif st.session_state.get("analyst") == "test1":
        st.info("โหมดทดสอบ 1: กำลังสุ่มข้อมูล")
        with st.spinner("กำลังสร้างข้อมูล..."):
            time.sleep(1)
            image_pil_test = Image.open(uploaded_file).convert("RGB")
            N3 = random.randint(10, 50)
            N2 = random.randint(20, 60)
            N1 = random.randint(5, 30)
            N0 = random.randint(0, 10)
            total_grains = N3 + N2 + N1 + N0
            numerator = (3*N3) + (2*N2) + (1*N1) + (0*N0)
            denominator = 3 * total_grains
            cri_score = (numerator / denominator) * 100 if denominator > 0 else 0
            
            st.session_state.analysis_results = {
                "N3": N3, "N2": N2, "N1": N1, "N0": N0,
                "total_grains": total_grains, "cri": cri_score,
                "processed_image": image_pil_test, "total_detected": total_grains + random.randint(5, 20)
            }
        st.success("✅ แสดงผลด้วยข้อมูลสุ่มสำเร็จ!")
        st.rerun()

    elif st.session_state.get("analyst") == "test2":
        st.info("โหมดทดสอบ 2: กำลังใช้ข้อมูลคงที่")
        with st.spinner("กำลังสร้างข้อมูล..."):
            time.sleep(1)
            image_pil_test = Image.open(uploaded_file).convert("RGB")
            st.session_state.analysis_results = {
                "N3": 17, "N2": 43, "N1": 26, "N0": 9, "total_grains": 95,
                "cri": 57.19, "processed_image": image_pil_test, "total_detected": 109 
            }
        st.success("✅ แสดงผลด้วยข้อมูลคงที่สำเร็จ!")
        st.rerun()

    else:
        try:
            with st.spinner("🧠 กำลังประมวลผลด้วย AI... (ขั้นตอนนี้อาจใช้เวลาสักครู่)"):
                confidence_threshold = 40 # กำหนดค่าตายตัวที่นี่ (40%)
                image_pil = Image.open(uploaded_file).convert("RGB")
                
                temp_path_detector = "temp_detector.jpg"
                image_pil.save(temp_path_detector)
                detections = detector_model.predict(temp_path_detector, confidence=5, overlap=30).json().get('predictions', [])
                os.remove(temp_path_detector)

                full_results = []
                temp_crop_dir = "temp_crops"
                os.makedirs(temp_crop_dir, exist_ok=True)
                
                progress_bar = st.progress(0, text="กำลังจำแนกประเภท...")
                for i, detection in enumerate(detections):
                    try:
                        x,y,w,h = float(detection.get('x',0)),float(detection.get('y',0)),float(detection.get('width',0)),float(detection.get('height',0))
                        if w <= 0 or h <= 0: continue
                        
                        cropped_img = image_pil.crop((int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)))
                        predicted_class, confidence = None, 0.0
                        try:
                            temp_predict_path = os.path.join(temp_crop_dir, f"predict_{i}.jpg")
                            cropped_img.save(temp_predict_path)
                            raw_result = classifier_model.predict(temp_predict_path, confidence=0).json()
                            if 'top' in raw_result and raw_result['top'] != "":
                                predicted_class = raw_result.get('top')
                                confidence = raw_result.get('confidence', 0.0)
                        except Exception:
                            pass
                        
                        full_results.append({"class": predicted_class, "confidence": confidence})
                    except Exception:
                        continue
                    progress_bar.progress((i + 1) / len(detections), text=f"จำแนกผลึกชิ้นที่ {i+1}/{len(detections)}")
                
                progress_bar.empty()
                if os.path.exists(temp_crop_dir):
                    shutil.rmtree(temp_crop_dir)

                confident_results = []
                for res in full_results:
                    if res.get('class') and (res.get('confidence', 0) * 100 >= confidence_threshold):
                        confident_results.append(res)
                
                classified_classes = [res['class'] for res in confident_results]
                total_grains = len(classified_classes)
                grade_counts = Counter(classified_classes)

                N3, N2, N1, N0 = grade_counts.get('class 3',0), grade_counts.get('class 2',0), grade_counts.get('class 1',0), grade_counts.get('class 0',0)
                
                cri_score = 0.0
                if total_grains > 0:
                     numerator = (3 * N3) + (2 * N2) + (1 * N1) + (0 * N0)
                     denominator = 3 * total_grains
                     cri_score = (numerator / denominator) * 100

                st.session_state.analysis_results = {
                    "N3": N3, "N2": N2, "N1": N1, "N0": N0,
                    "total_grains": total_grains,
                    "total_detected": len(detections),
                    "cri": cri_score,
                    "processed_image": image_pil
                }

            st.success("✅ ประมวลผลสำเร็จ!")
            st.info("คุณสามารถดูการแสดงผลด้วยกราฟได้ที่หน้า '📊 Graph Analysis' จากเมนูด้านซ้าย (ถ้ามี)")
        except Exception as e:
            st.error(f"😭 เกิดข้อผิดพลาดร้ายแรงระหว่างการประมวลผล!")
            st.exception(e)
        
        st.rerun()

with col_main_right:
    results = st.session_state.analysis_results

    if results:
        st.image(results["processed_image"], caption="ภาพที่วิเคราะห์", use_container_width=True)
    elif uploaded_file:
        st.image(uploaded_file, caption="ภาพที่เลือก (รอการประมวลผล)", use_container_width=True)
    else:
        placeholder_image = np.full((400, 600, 3), 240, dtype=np.uint8)
        cv2.putText(placeholder_image, "Upload an image to start", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        st.image(placeholder_image, caption="ผลการวิเคราะห์จะแสดงที่นี่", use_container_width=True)

    if results:
        total_detected = results.get("total_detected", 0)
        total_classified_confidently = results.get("total_grains", 0)
        st.info(f"AI ตรวจพบผลึก **{total_detected}** ชิ้น และจำแนกได้อย่างมั่นใจ **{total_classified_confidently}** ชิ้น")

    st.markdown("### ผลการประเมินคะแนนเม็ดน้ำตาล")

    N3 = results.get("N3", 0) if results else 0
    N2 = results.get("N2", 0) if results else 0
    N1 = results.get("N1", 0) if results else 0
    N0 = results.get("N0", 0) if results else 0
    total_grains = results.get("total_grains", 0) if results else 0
    cri = results.get("cri", 0.0) if results else 0
    
    p3 = (N3 / total_grains * 100) if total_grains > 0 else 0
    p2 = (N2 / total_grains * 100) if total_grains > 0 else 0
    p1 = (N1 / total_grains * 100) if total_grains > 0 else 0
    p0 = (N0 / total_grains * 100) if total_grains > 0 else 0
    
    data_to_display = [{"class": 3, "count": N3, "percent": p3}, {"class": 2, "count": N2, "percent": p2},
                       {"class": 1, "count": N1, "percent": p1}, {"class": 0, "count": N0, "percent": p0}]

    for item in data_to_display:
        g, c, m, p, pct = st.columns([1, 1, 0.5, 1, 0.5])
        with g: st.markdown(f"คะแนน {item['class']}")
        with c: st.text_input("", str(item['count']), disabled=True, key=f"d_c_{item['class']}", label_visibility="collapsed")
        with m: st.markdown("เม็ด")
        with p: st.text_input("", f"{item['percent']:.2f}", disabled=True, key=f"d_p_{item['class']}", label_visibility="collapsed")
        with pct: st.markdown("%")

    t1, t2, t3, t4, t5 = st.columns([1, 1, 0.5, 1, 0.5])
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
                record_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_date": st.session_state.get("date_input").strftime("%Y-%m-%d"),
                    "strike": st.session_state.get("stike"),
                    "class_selection": st.session_state.get("class_selection"),
                    "analyst": st.session_state.get("analyst"),
                    "operator": st.session_state.get("operator"),
                    "total_detected": results.get("total_detected", 0),
                    "total_classified": results.get("total_grains", 0),
                    "class_3_count": results.get("N3", 0),
                    "class_2_count": results.get("N2", 0),
                    "class_1_count": results.get("N1", 0),
                    "class_0_count": results.get("N0", 0),
                    "cri_score": f"{results.get('cri', 0.0):.2f}"
                }
                save_results_to_csv(record_data)
                st.success(f"✅ บันทึกข้อมูลเรียบร้อยแล้ว!")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการบันทึก: {e}")
        else:
            st.error("❌ ไม่มีข้อมูลให้บันทึก")

    CSV_FILENAME = "analysis_history.csv"
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "rb") as f:
            st.download_button(
                label="📥 ดาวน์โหลดประวัติการวิเคราะห์ (CSV)",
                data=f,
                file_name=CSV_FILENAME,
                mime="text/csv",
                key="download_csv_button"
            )

# =================================================================
# 4. ส่วนแสดงกราฟ (ย้ายมาไว้ที่ฝั่งซ้าย)
# =================================================================
with col_main_left:
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("กราฟสรุปจำนวนผลึก")
        
        results_for_graph = st.session_state.analysis_results
        
        chart_data = {
            'Class': ['Class 3', 'Class 2', 'Class 1', 'Class 0'],
            'Count': [
                results_for_graph.get("N3", 0), results_for_graph.get("N2", 0),
                results_for_graph.get("N1", 0), results_for_graph.get("N0", 0)
            ]
        }
        df_chart = pd.DataFrame(chart_data)

        fig_bar = px.bar(
            df_chart[df_chart['Count'] > 0],
            x='Class', y='Count', color='Class', text_auto=True,
            color_discrete_map={
                'Class 3': '#4CAF50', 'Class 2': '#8BC34A',
                'Class 1': '#FFC107', 'Class 0': '#F44336'
            },
            labels={'Count': 'จำนวนเม็ด', 'Class': ''}
        )
        fig_bar.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

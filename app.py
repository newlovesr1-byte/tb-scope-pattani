import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="TB-SCOPE AI | โรงพยาบาลปัตตานี", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ตกแต่ง CSS สไตล์โรงพยาบาล
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0056b3 0%, #008080 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #f0f2f6;
        margin-top: 5px;
    }
    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #0056b3;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัวของเว็บ
st.markdown("""
    <div class="main-header">
        <h1>🫁 TB-SCOPE AI: ระบบคัดกรองความเสี่ยงวัณโรค</h1>
        <div class="sub-header">งานอณูชีวโมเลกุล กลุ่มงานเทคนิคการแพทย์และพยาธิวิทยาคลินิก โรงพยาบาลปัตตานี</div>
    </div>
""", unsafe_allow_html=True)

# 3. โหลดโมเดลและไฟล์ที่เซฟไว้
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('tb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features_list.pkl')
        return model, scaler, features, True
    except:
        return None, None, None, False

model, scaler, features, model_loaded = load_artifacts()

if not model_loaded:
    st.warning("⚠️ **โหมดจำลอง:** ยังไม่พบไฟล์โมเดล (`tb_model.pkl`) ระบบจะใช้ค่าจำลองในการแสดงผล กรุณานำไฟล์โมเดลมาวางรวมกับโค้ดเพื่อใช้งานจริง")

# 4. ฟอร์มรับข้อมูลผู้ป่วย (Sidebar)
st.sidebar.markdown("### 📝 บันทึกข้อมูลผู้ป่วย")
st.sidebar.markdown("กรอกผลตรวจทางห้องปฏิบัติการ (CBC) และอายุ")

with st.sidebar.form("patient_form"):
    st.markdown("#### ข้อมูลพื้นฐานและโลหิตวิทยา")
    age = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=50)
    wbc = st.number_input("WBC (10^3/uL)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    rbc = st.number_input("RBC (10^6/uL)", min_value=0.0, max_value=10.0, value=4.5, step=0.01)
    hb = st.number_input("Hb (g/dL)", min_value=0.0, max_value=25.0, value=12.0, step=0.1)
    hct = st.number_input("Hct (%)", min_value=0.0, max_value=70.0, value=36.0, step=0.1)
    mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=120.0, value=85.0, step=0.1)
    rdw = st.number_input("RDW (%)", min_value=0.0, max_value=30.0, value=14.0, step=0.1)
    plt = st.number_input("PLT (10^3/uL)", min_value=0, max_value=1000, value=250)
    
    st.markdown("#### Differential WBC (%)")
    neu = st.number_input("Neutrophil (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    lym = st.number_input("Lymphocyte (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    mono = st.number_input("Monocyte (%)", min_value=0.0, max_value=100.0, value=6.0, step=0.1)
    
    submitted = st.form_submit_button("🔍 ประเมินความเสี่ยงวัณโรค", use_container_width=True)

# 5. การแสดงผลและประมวลผลหลัก
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown("#### 📊 สรุปข้อมูลที่ป้อนเข้าสู่ระบบ")
    data = {
        "Age": [age], "WBC": [wbc], "RBC": [rbc], "Hb": [hb], "Hct": [hct],
        "MCV": [mcv], "RDW": [rdw], "PLT": [plt], "Neutrophil": [neu],
        "Lymphocyte": [lym], "Monocyte": [mono]
    }
    input_df = pd.DataFrame(data)
    
    # คำนวณอัตราส่วนการอักเสบอัตโนมัติ
    input_df["ANC"] = input_df["WBC"] * input_df["Neutrophil"] / 100
    input_df["ALC"] = input_df["WBC"] * input_df["Lymphocyte"] / 100
    input_df["AMC"] = input_df["WBC"] * input_df["Monocyte"] / 100
    input_df["NLR"] = input_df["ANC"] / input_df["ALC"]
    input_df["PLR"] = input_df["PLT"] / input_df["ALC"]
    input_df["MLR"] = input_df["AMC"] / input_df["ALC"]
    input_df["SII"] = input_df["PLT"] * input_df["ANC"] / input_df["ALC"]
    
    st.dataframe(input_df.T.rename(columns={0: "ค่าที่กรอก"}), use_container_width=True)

with col2:
    st.markdown("#### 🎯 ผลการวิเคราะห์ความเสี่ยง")
    
    if submitted:
        with st.spinner("กำลังประมวลผลด้วยโมเดล Machine Learning..."):
            if model_loaded:
                X_input = input_df[features]
                X_scaled = scaler.transform(X_input)
                prob = model.predict_proba(X_scaled)[0][1]
            else:
                prob = 0.785
            
            threshold = 0.495 # Optimal Threshold
            
            st.markdown(f"""
                <div class="card">
                    <h3>ความน่าจะเป็น (Probability)</h3>
                    <h1 style="color: {'#d9534f' if prob >= threshold else '#5cb85c'};">{prob*100:.1f}%</h1>
                    <p>เกณฑ์ตัดสินใจ (Threshold): {threshold*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            if prob >= threshold:
                st.error("⚠️ **คำแนะนำทางคลินิก:** ผลการประเมินอยู่ใน **กลุ่มเสี่ยงสูง**\n\n*แนะนำ:* ควรพิจารณาส่งตรวจยืนยันทางอณูชีวโมเลกุล (Xpert MTB/RIF) เพิ่มเติมตามดุลยพินิจของแพทย์")
            else:
                st.success("✅ **คำแนะนำทางคลินิก:** ผลการประเมินอยู่ใน **กลุ่มเสี่ยงต่ำ**\n\n*แนะนำ:* โอกาสติดเชื้อต่ำกว่าเกณฑ์คัดกรอง แต่อาจพิจารณาติดตามอาการตามบริบททางคลินิก")
    else:
        st.info("👈 กรุณากรอกข้อมูลผลเลือดทางด้านซ้าย แล้วกดปุ่ม **'ประเมินความเสี่ยงวัณโรค'**")

# ส่วนท้ายเว็บไซต์
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.85rem;'>"
    "พัฒนาและดูแลโดย งานอณูชีวโมเลกุล กลุ่มงานเทคนิคการแพทย์และพยาธิวิทยาคลินิก โรงพยาบาลปัตตานี<br>"
    "ระบบนี้ใช้เพื่อเป็นเครื่องมือสนับสนุนการตัดสินใจเบื้องต้น (Clinical Decision Support System) เท่านั้น"
    "</p>", 
    unsafe_allow_html=True
)
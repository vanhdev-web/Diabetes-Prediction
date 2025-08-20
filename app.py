import streamlit as st
import numpy as np
import pickle

# load model và scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Prediction App")
st.write("Nhập các thông tin dưới đây để dự đoán nguy cơ mắc bệnh tiểu đường:")

# hàm reset: gán lại giá trị mặc định cho session_state
def reset_form():
    defaults = {
        "pregnancies": 0,
        "glucose": 0,
        "blood_pressure": 0,
        "skin_thickness": 0,
        "insulin": 0,
        "bmi": 0.0,
        "dpf": 0.0,
        "age": 1
    }
    for key, value in defaults.items():
        st.session_state[key] = value

# tạo form
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, key="pregnancies")
        glucose = st.slider("Glucose", min_value=0, max_value=300, value=st.session_state.get("glucose",0), key="glucose")
        blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=200, value=st.session_state.get("blood_pressure",0), key="blood_pressure")
        skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, value=st.session_state.get("skin_thickness",0), key="skin_thickness")

    with col2:
        insulin = st.slider("Insulin", min_value=0, max_value=900, value=st.session_state.get("insulin",0), key="insulin")
        bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=st.session_state.get("bmi",0.0), format="%.1f", key="bmi")
        dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=st.session_state.get("dpf",0.0), format="%.3f", key="dpf")
        age = st.number_input("Age", min_value=1, max_value=120, step=1, key="age")

    # 2 nút sát nhau


    submit_button = st.form_submit_button(label="Dự đoán")
    reset_button = st.form_submit_button(label="Thử lại", on_click=reset_form)

if submit_button:
    input_data = np.array([[st.session_state["pregnancies"],
                            st.session_state["glucose"],
                            st.session_state["blood_pressure"],
                            st.session_state["skin_thickness"],
                            st.session_state["insulin"],
                            st.session_state["bmi"],
                            st.session_state["dpf"],
                            st.session_state["age"]]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.error("Kết quả: Có nguy cơ mắc tiểu đường.")
    else:
        st.success("Kết quả: Không có nguy cơ mắc tiểu đường.")

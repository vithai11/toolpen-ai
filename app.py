import streamlit as st
import random
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="⚽ AI Dự Đoán Penalty", page_icon="⚽")

# Initialize session states
if 'kick_history' not in st.session_state:
    st.session_state.kick_history = []  # lưu 3 lượt gần nhất
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_enc' not in st.session_state:
    st.session_state.label_enc = LabelEncoder()

# Tạo fake dữ liệu ban đầu để model không lỗi
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['kick', 'goalkeeper', 'result'])

# Giao diện
st.title("1. Bạn chọn hướng sút:")
kick = st.radio("", ['Trái', 'Giữa', 'Phải'])

st.title("2. Thủ môn nhảy hướng nào?")
goalkeeper = st.radio("", ['Trái', 'Giữa', 'Phải'])

# Xử lý kết quả
result = "Ghi bàn" if kick != goalkeeper else "Bị cản phá"

# Lưu lịch sử
st.session_state.kick_history.append((kick, goalkeeper, result))
if len(st.session_state.kick_history) > 3:
    st.session_state.kick_history.pop(0)

# Lưu vào dataframe học
new_data = pd.DataFrame([[kick, goalkeeper, result]], columns=['kick', 'goalkeeper', 'result'])
st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)

st.success(f"{result}! (Sút: {kick}, Thủ môn: {goalkeeper})")

# Encode dữ liệu
def encode_data(df):
    df_encoded = df.copy()
    for col in ['kick', 'goalkeeper', 'result']:
        df_encoded[col] = st.session_state.label_enc.fit_transform(df[col])
    return df_encoded

# Train model nếu có ít nhất 5 dòng
if len(st.session_state.df) >= 5:
    data = encode_data(st.session_state.df)
    X = data[['goalkeeper']]
    y = data['kick']
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    st.session_state.model = model

# Dự đoán lượt tiếp theo
if st.session_state.model:
    current_goalkeeper_move = st.selectbox("Dự đoán thủ môn sẽ nhảy hướng nào?", ['Trái', 'Giữa', 'Phải'])
    move_encoded = st.session_state.label_enc.transform([current_goalkeeper_move])[0]
    
    pred = st.session_state.model.predict([[move_encoded]])[0]
    pred_label = st.session_state.label_enc.inverse_transform([pred])[0]
    
    st.header("Gợi ý từ AI (cho lượt kế tiếp):")
    st.success(f"Nên sút về: {pred_label}")
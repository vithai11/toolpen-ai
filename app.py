import streamlit as st
import random
import numpy as np
from collections import deque
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config(page_title="AI Dự đoán hướng sút", layout="centered")
st.title("⚽ AI Dự Đoán Hướng Sút - XGBoost & Tự Học Theo 3 Lượt Gần Nhất")

# Khởi tạo session
if "kick_history" not in st.session_state:
    st.session_state.kick_history = deque(maxlen=100)
    st.session_state.goalie_jump_history = deque(maxlen=100)
    st.session_state.result = ""
    st.session_state.encoder = LabelEncoder()
    st.session_state.encoder.fit(["left", "center", "right"])
    st.session_state.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    st.session_state.success_count = 0
    st.session_state.total_shots = 0
    st.session_state.ai_suggestion = ""
    st.session_state.prediction_result = ""
    st.session_state.pending_direction = None
    st.session_state.last_probs = None

# Gợi ý từ AI
def smart_kick_xgb():
    if len(st.session_state.kick_history) < 4:
        return random.choice(["left", "center", "right"]), None

    try:
        kicks = list(st.session_state.kick_history)[-3:]
        jumps = list(st.session_state.goalie_jump_history)[-3:]
        if len(kicks) < 3 or len(jumps) < 3:
            return random.choice(["left", "center", "right"]), None

        input_feature = [kicks + jumps]
        probs = st.session_state.model.predict_proba(input_feature)[0]
        max_index = np.argmax(probs)
        likely_jump = st.session_state.encoder.inverse_transform([max_index])[0]
        options = {"left", "center", "right"} - {likely_jump}
        return random.choice(list(options)), probs
    except Exception as e:
        return random.choice(["left", "center", "right"]), None

# B1: Chọn hướng sút
st.markdown("### 1. Bạn chọn hướng sút:")
cols = st.columns(3)
direction = None
if cols[0].button("⬅️ Trái"):
    direction = "left"
if cols[1].button("⬆️ Giữa"):
    direction = "center"
if cols[2].button("➡️ Phải"):
    direction = "right"

if direction:
    st.session_state.pending_direction = direction

# B2: Hướng thủ môn nhảy
if st.session_state.pending_direction:
    st.markdown("### 2. Thủ môn nhảy hướng nào?")
    gcols = st.columns(3)
    goalie_dir = None
    if gcols[0].button("🧤 Trái"):
        goalie_dir = "left"
    if gcols[1].button("🧤 Giữa"):
        goalie_dir = "center"
    if gcols[2].button("🧤 Phải"):
        goalie_dir = "right"

    if goalie_dir:
        kick_num = st.session_state.encoder.transform([st.session_state.pending_direction])[0]
        goalie_num = st.session_state.encoder.transform([goalie_dir])[0]
        st.session_state.kick_history.append(kick_num)
        st.session_state.goalie_jump_history.append(goalie_num)

        # Huấn luyện nếu đủ data
        if len(st.session_state.kick_history) >= 6:
            X, y = [], []
            for i in range(3, len(st.session_state.kick_history)):
                X.append(
                    list(st.session_state.kick_history)[i-3:i] +
                    list(st.session_state.goalie_jump_history)[i-3:i]
                )
                y.append(st.session_state.goalie_jump_history[i])
            st.session_state.model.fit(X, y)

        # Kết quả
        if st.session_state.pending_direction == goalie_dir:
            st.session_state.result = f"❌ Bị bắt! Thủ môn nhảy đúng hướng: {goalie_dir}"
            st.session_state.prediction_result = "Sai dự đoán"
        else:
            st.session_state.result = f"✅ Ghi bàn! Thủ môn nhảy sang {goalie_dir}"
            st.session_state.success_count += 1
            st.session_state.prediction_result = "Đúng dự đoán"

        st.session_state.total_shots += 1

        # Gợi ý
        st.session_state.ai_suggestion, st.session_state.last_probs = smart_kick_xgb()
        st.session_state.pending_direction = None

# Hiển thị kết quả lượt chơi
if st.session_state.result:
    st.info(st.session_state.result)
    st.markdown(f"**Dự đoán của AI:** {st.session_state.prediction_result}")

# Gợi ý AI
st.markdown("---")
st.subheader("Gợi ý từ AI (cho lượt kế tiếp):")
if st.session_state.ai_suggestion:
    st.success(f"**Nên sút về: {st.session_state.ai_suggestion.upper()}**")
    if st.session_state.last_probs is not None:
        st.write(f"**Xác suất thủ môn nhảy:**")
        st.write(f"🔹 Trái: {st.session_state.last_probs[0] * 100:.2f}%")
        st.write(f"🔹 Giữa: {st.session_state.last_probs[1] * 100:.2f}%")
        st.write(f"🔹 Phải: {st.session_state.last_probs[2] * 100:.2f}%")

# Tỷ lệ thành công
if st.session_state.total_shots > 0:
    acc = 100 * st.session_state.success_count / st.session_state.total_shots
    st.markdown(f"**Tỷ lệ sút thành công:** `{acc:.2f}%`")

# Lịch sử
st.markdown("### Lịch sử lượt chơi:")
if st.session_state.kick_history:
    st.write("Hướng sút:", list(st.session_state.encoder.inverse_transform(st.session_state.kick_history)))
    st.write("Thủ môn nhảy:", list(st.session_state.encoder.inverse_transform(st.session_state.goalie_jump_history)))
else:
    st.write("*Chưa có dữ liệu.*")

# Reset
if st.button("🔄 Reset game"):
    st.session_state.kick_history.clear()
    st.session_state.goalie_jump_history.clear()
    st.session_state.result = ""
    st.session_state.success_count = 0
    st.session_state.total_shots = 0
    st.session_state.ai_suggestion = ""
    st.session_state.prediction_result = ""
    st.session_state.pending_direction = None
    st.session_state.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
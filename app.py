import streamlit as st
import cv2
import PIL.Image
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import mediapipe as mp
import tempfile
import os
from datetime import datetime
import io
import zipfile
import math

# ---------------------------------------------------------
# 1. API 설정
# ---------------------------------------------------------
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    # 로컬 테스트용 (배포 시에는 secrets가 우선됨)
    GOOGLE_API_KEY = "AIzaSyANlIKJWsIon4JbrR2U-WUosLkfGts8PYs"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash') 
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
RED_STYLE = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3)
YELLOW_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=4)

# 안전 설정
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ---------------------------------------------------------
# 2. 수학적 계산 함수
# ---------------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return int(angle)

# ---------------------------------------------------------
# 3. 핵심 분석 함수
# ---------------------------------------------------------
def extract_frames_from_video_file(video_path, num_frames=10):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened(): return []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return []
    step = total_frames // (num_frames + 1)
    pil_images = []
    for i in range(1, num_frames + 1):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, image = vidcap.read()
        if success:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_images.append(PIL.Image.fromarray(image_rgb))
    vidcap.release()
    return pil_images

def analyze_pose_and_draw(pil_image):
    image_np = np.array(pil_image)
    height, width, _ = image_np.shape

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        results = pose.process(image_np)
        annotated_image = image_np.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=RED_STYLE, connection_drawing_spec=RED_STYLE
            )

            # 각도 계산 로직
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
            
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]

            angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            angle_r = calculate_angle(hip_r, knee_r, ankle_r)

            cv2.putText(annotated_image, f"{angle_l}", tuple(np.multiply(knee_l, [1, 1]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated_image, f"{angle_r}", tuple(np.multiply(knee_r, [1, 1]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

            # 후경 체크
            mid_hip_x = (hip_l[0] + hip_r[0]) / 2
            mid_ankle_x = (ankle_l[0] + ankle_r[0]) / 2
            
            cv2.line(annotated_image, (int(mid_hip_x), int(hip_l[1])), (int(mid_hip_x), int(ankle_l[1])), (255, 255, 0), 2)
            
            if angle_l > 165 or angle_r > 165:
                 cv2.putText(annotated_image, "HIGH!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

    return PIL.Image.fromarray(annotated_image)

def generate_feedback_with_captions(frames, student_name, coach_name, weekly_goal, morning_note, afternoon_note):
    today_date = datetime.now().strftime("%m/%d")
    frame_count = len(frames)
    
    if not morning_note: morning_note = "기초 훈련 진행"
    if not afternoon_note: afternoon_note = "심화 훈련 진행"

    prompt = f"""
    당신은 스키 팀의 담당 코치 '{coach_name}'입니다.
    제공된 {frame_count}장의 사진은 수강생 '{student_name}'님의 스키 타는 모습입니다.
    사진에는 **무릎 각도(숫자)**와 **자세 뼈대**가 그려져 있습니다. 이 데이터를 참고하여 분석해 주세요.
    
    [핵심 지시사항]
    아래 '코치가 입력한 훈련 내용'을 바탕으로 피드백을 작성해 주세요.
    입력 내용을 최대한 반영하되, 문장을 최신 스키 용어와 전문적인 '해요체'로 다듬어 주세요.
    
    [입력 정보]
    - 회원: {student_name}
    - 일자: {today_date}
    - 담당: {coach_name} 코치
    - 주간 목표: {weekly_goal}
    
    [코치가 입력한 훈련 내용]
    - 오전 포인트: {morning_note}
    - 오후 포인트: {afternoon_note}

    [출력 양식]
    두 가지 파트로 나누고, 사이에는 '|||' (파이프 3개)를 넣어주세요.

    [PART 1: 학부모 전송용 피드백]
    {student_name} - ⛷ 비바365 레슨 피드백

    ✪ 코드 : S 클래스
    ✪ 회원 : {student_name}
    ✪ 일자 : {today_date}
    ✪ 담당 : {coach_name} 코치

    ∎ 주간 교육과정 및 목표
    {weekly_goal}

    📌오전 : 
    (오전 포인트 내용을 바탕으로 3줄 작성)

    📌 오후 교육 : 
    (오후 포인트 내용을 바탕으로 3줄 작성)

    (칭찬 멘트)

    시즌반 S클래스의 피드백은 매일 전달되는 방식이 아니라,
    아이들의 발전 단계와 필요에 따라 수시로 제공되고 있습니다.
    
    조금만 믿고 지켜봐 주시면, 더 큰 성장의 감동을 전달 드리겠습니다.😊

    
    [PART 2: 사진별 분석]
    각 사진(1번~{frame_count}번)에 대해 '체크 포인트'를 한 문장으로 작성.
    구분자는 '###' 사용.
    """
    
    response = model.generate_content([prompt, *frames], safety_settings=safety_settings)
    if response.parts: return response.text
    else: return "분석 실패 ||| 분석 실패 ### 분석 실패"

def create_zip_file(images, selected_indices):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for idx in selected_indices:
            img = images[idx]
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            zf.writestr(f"ski_scene_{idx+1}.jpg", img_byte_arr.getvalue())
    return zip_buffer.getvalue()

# ---------------------------------------------------------
# 4. 메인 웹 화면
# ---------------------------------------------------------
st.set_page_config(page_title="AI 스키 코치 Pro", page_icon="⛷️")
st.title("⛷️ 스키 정밀 분석기 (Pro Ver.)")
st.caption("AI Vision + 수학적 각도 계산이 포함된 버전입니다.")

if 'analyzed_images' not in st.session_state:
    st.session_state.analyzed_images = []
if 'captions' not in st.session_state:
    st.session_state.captions = []
if 'main_feedback' not in st.session_state:
    st.session_state.main_feedback = ""

with st.form("feedback_form"):
    uploaded_file = st.file_uploader("동영상 업로드", type=['mp4', 'mov'])
    col1, col2 = st.columns(2)
    with col1: student_name = st.text_input("회원 이름", placeholder="김승후")
    with col2: coach_name = st.text_input("담당 코치", placeholder="신정우")
    weekly_goal = st.text_input("주간 교육 목표", placeholder="패러렐 턴")
    
    col_am, col_pm = st.columns(2)
    with col_am: morning_note = st.text_area("📌 오전 교육 내용", height=80)
    with col_pm: afternoon_note = st.text_area("📌 오후 교육 내용", height=80)
    
    submitted = st.form_submit_button("🚀 정밀 분석 시작")

if submitted:
    if not uploaded_file:
        st.warning("영상을 먼저 업로드해주세요.")
    elif GOOGLE_API_KEY == "여기에_발급받은_API_키를_넣으세요":
        st.error("API 키 확인 필요")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        try:
            with st.spinner("1단계: 관절 각도 계산 및 시각화 중..."):
                raw_frames = extract_frames_from_video_file(video_path, num_frames=10)
                processed_frames = []
                for frame in raw_frames:
                    processed_frames.append(analyze_pose_and_draw(frame))
                st.session_state.analyzed_images = processed_frames

            if raw_frames:
                with st.spinner("2단계: 데이터 기반 AI 리포트 생성 중..."):
                    full_response = generate_feedback_with_captions(
                        raw_frames, student_name, coach_name, weekly_goal, morning_note, afternoon_note
                    )
                    try:
                        parts = full_response.split("|||")
                        st.session_state.main_feedback = parts[0].strip()
                        if len(parts) > 1: st.session_state.captions = parts[1].strip().split("###")
                        else: st.session_state.captions = ["분석 내용 없음"] * len(raw_frames)
                    except:
                        st.session_state.main_feedback = full_response
                        st.session_state.captions = ["오류"] * len(raw_frames)
            else:
                st.error("영상 처리 실패")
        except Exception as e:
            st.error(f"오류: {e}")
        finally:
            if os.path.exists(video_path): os.unlink(video_path)

if st.session_state.analyzed_images:
    st.divider()
    st.subheader(f"📸 정밀 분석 결과 (총 {len(st.session_state.analyzed_images)}장)")
    st.info("무릎 옆의 노란색 숫자는 '관절의 각도'입니다.")

    selected_indices = []
    # -------------------------------------------------------------
    # [수정됨] use_container_width=True -> width="stretch" 로 변경
    # -------------------------------------------------------------
    for i in range(0, len(st.session_state.analyzed_images), 2):
        cols = st.columns(2)
        with cols[0]:
            if i < len(st.session_state.analyzed_images):
                st.image(st.session_state.analyzed_images[i], width="stretch")
                caption = st.session_state.captions[i].strip() if i < len(st.session_state.captions) else ""
                st.info(f"{i+1}. {caption}")
                if st.checkbox(f"선택 {i+1}", key=f"c{i}"): selected_indices.append(i)
        with cols[1]:
            if i+1 < len(st.session_state.analyzed_images):
                st.image(st.session_state.analyzed_images[i+1], width="stretch")
                caption = st.session_state.captions[i+1].strip() if i+1 < len(st.session_state.captions) else ""
                st.info(f"{i+2}. {caption}")
                if st.checkbox(f"선택 {i+2}", key=f"c{i+1}"): selected_indices.append(i+1)

    st.markdown("---")
    if selected_indices:
        zip_data = create_zip_file(st.session_state.analyzed_images, selected_indices)
        st.download_button("📦 선택한 사진 다운로드 (ZIP)", data=zip_data, file_name=f"{student_name}_analysis.zip", mime="application/zip", type="primary")

    st.divider()
    st.subheader("📝 피드백 리포트")
    st.text_area("카톡 전송용", st.session_state.main_feedback, height=350)
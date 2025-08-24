import streamlit as st
import cv2
import os
import pickle
import face_recognition
import pandas as pd
from datetime import datetime

# --------------------------
# Paths
# --------------------------
OUTPUT_DIR = "output"
FACES_DIR = "data/faces"
GROUP_PHOTOS_DIR = "data/group_photos"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(GROUP_PHOTOS_DIR, exist_ok=True)

# --------------------------
# Save / Load encodings
# --------------------------
def save_encodings(encodings, path=os.path.join(OUTPUT_DIR, "encodings.pkl")):
    with open(path, "wb") as f:
        pickle.dump(encodings, f)
    st.success(f" Encodings saved to {path}")

def load_encodings(path=os.path.join(OUTPUT_DIR, "encodings.pkl")):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

# --------------------------
# Live capture photos
# --------------------------
def capture_photos(reg_no, name, branch, session, num_photos=5):
    folder_name = f"{reg_no}_{name.replace(' ', '_')}_{branch.replace(' ', '')}_{session.replace(' ', '')}"
    folder_path = os.path.join(FACES_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    frame_placeholder = st.empty()

    st.info(" Capturing photos... Please stay still and look at the camera")

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected ")
            break

        frame_placeholder.image(frame, channels="BGR", caption=f"Capturing Photo {count+1}/{num_photos}")
        file_path = os.path.join(folder_path, f"{count+1}.jpg")
        cv2.imwrite(file_path, frame)
        count += 1

    cap.release()
    st.success(f"  {num_photos} photos captured and saved to {folder_path}")

# --------------------------
# Encode known faces (with progress bar)
# --------------------------
def encode_faces():
    encodings = {}
    student_dirs = [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]

    progress_bar = st.progress(0)       # progress bar
    status_text = st.empty()            # status message
    total_students = len(student_dirs)

    for idx, student_dir in enumerate(student_dirs, start=1):
        student_path = os.path.join(FACES_DIR, student_dir)

        student_encodings = []
        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    student_encodings.append(face_encs[0])
            except Exception as e:
                st.warning(f" Skipping {img_path}: {e}")

        if student_encodings:
            encodings[student_dir] = student_encodings

        # update progress
        progress = int((idx / total_students) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Encoding {idx}/{total_students} students...")

    save_encodings(encodings)
    st.success(" Training Completed!")
    return encodings

# --------------------------
# Parse student folder name
# --------------------------
def parse_student(student_key: str):
    parts = student_key.split("_")
    if len(parts) >= 4:
        reg_no = parts[0]
        branch = parts[-2]
        session = parts[-1]
        name = "_".join(parts[1:-2])
    elif len(parts) >= 2:
        reg_no = parts[0]
        name = "_".join(parts[1:])
        branch = "Unknown"
        session = "Unknown"
    else:
        reg_no, name, branch, session = student_key, "Unknown", "Unknown", "Unknown"

    name = name.replace("_", " ")
    return reg_no, name, branch, session

# --------------------------
# Mark Attendance
# --------------------------
def mark_attendance(group_photo_path, encodings, selected_branch, selected_session):
    cohort_keys = []
    for k in encodings.keys():
        _, _, b, s = parse_student(k)
        if b == selected_branch and s == selected_session:
            cohort_keys.append(k)

    if not cohort_keys:
        st.warning(" No students found for this Branch & Session.")
        return

    group_image = face_recognition.load_image_file(group_photo_path)
    group_face_locations = face_recognition.face_locations(group_image)
    group_face_encodings = face_recognition.face_encodings(group_image, group_face_locations)

    st.info(f" Total faces detected in group photo: {len(group_face_encodings)}")

    attendance_rows = []
    matched_students = set()

    for face_enc in group_face_encodings:
        match_found = False
        for student_key in cohort_keys:
            results = face_recognition.compare_faces(encodings[student_key], face_enc, tolerance=0.5)
            if True in results:
                if student_key not in matched_students:
                    reg_no, name, branch, session = parse_student(student_key)
                    st.success(f" Match found: {reg_no} {name} ({branch}, {session})")
                    attendance_rows.append((reg_no, name, branch, session, "Present"))
                    matched_students.add(student_key)
                match_found = True
                break
        if not match_found:
            st.warning(" No match for a face in the group photo")

    for student_key in cohort_keys:
        if student_key not in matched_students:
            reg_no, name, branch, session = parse_student(student_key)
            attendance_rows.append((reg_no, name, branch, session, "Absent"))

    df = pd.DataFrame(attendance_rows, columns=["RegNo", "Name", "Branch", "Session", "Status"])
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.sort_values(by=["RegNo", "Name"], inplace=True)

    safe_branch = selected_branch.replace("/", "-")
    safe_session = selected_session.replace("/", "-")
    out_path = os.path.join(OUTPUT_DIR, f"attendance_{safe_branch}_{safe_session}.csv")
    df.to_csv(out_path, index=False)

    st.success(f"Attendance marked in {out_path}")
    st.dataframe(df)

# --------------------------
# Streamlit UI
# --------------------------
st.title(" Face Recognition Attendance System")

menu = st.sidebar.selectbox("Menu", ["Register Student", "Train Faces", "Upload Group Photo & Mark Attendance"])

if menu == "Register Student":
    st.subheader(" Register New Student")
    reg_no = st.text_input("Registration Number")
    name = st.text_input("Full Name")
    branch = st.selectbox("Branch", ["CSE(AI&ML)", "CSE(DS)", "CSE(Core)", "ECE", "EEE"])
    session = st.selectbox("Session", ["2022-2026", "2023-2027", "2024-2028", "2025-2029"])
    num_photos = st.slider("Number of photos to capture", 3, 15, 5)

    if st.button("Capture Photos"):
        if reg_no and name and branch and session:
            capture_photos(reg_no, name, branch, session, num_photos)
        else:
            st.error(" Please fill all fields")

elif menu == "Train Faces":
    st.subheader(" Train Faces and Generate Encodings")
    if st.button("Start Training"):
        encode_faces()

elif menu == "Upload Group Photo & Mark Attendance":
    st.subheader(" Upload Group Photo")
    uploaded_file = st.file_uploader("Upload a group photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        group_photo_path = os.path.join(GROUP_PHOTOS_DIR, uploaded_file.name)
        with open(group_photo_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(group_photo_path, caption="Uploaded Group Photo")

        encodings = load_encodings()
        if encodings:
            branch_options = ["CSE(AI&ML)", "CSE(DS)", "CSE(Core)", "ECE", "EEE"]
            session_options = ["2022-2026", "2023-2027", "2024-2028", "2025-2029"]

            selected_branch = st.selectbox("Select Branch", branch_options)
            selected_session = st.selectbox("Select Session", session_options)

            if st.button("Mark Attendance"):
                mark_attendance(group_photo_path, encodings, selected_branch, selected_session)
        else:
            st.error(" No encodings found. Please train faces first.")

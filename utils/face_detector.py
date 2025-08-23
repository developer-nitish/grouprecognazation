# import face_recognition
# import cv2
# import os
# import pandas as pd
# from datetime import datetime
# import pickle

# def mark_attendance(group_photo_path, encoding_file='output/encodings.pkl', output_csv='output/attendance.csv'):
#     with open(encoding_file, 'rb') as f:
#         known_encodings, metadata = pickle.load(f)

#     known_reg = [meta['reg_no'] for meta in metadata]
#     known_names = [meta['name'] for meta in metadata]

#     image = face_recognition.load_image_file(group_photo_path)
#     face_locations = face_recognition.face_locations(image)
#     face_encodings = face_recognition.face_encodings(image, face_locations)

#     present_reg = set()
#     present_name = set()

#     for encoding in face_encodings:
#         matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
#         if True in matches:
#             matched_idx = matches.index(True)
#             present_reg.add(metadata[matched_idx]['reg_no'])
#             present_name.add(metadata[matched_idx]['name'])

#     all_data = []
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     unique_students = {(meta['reg_no'], meta['name']) for meta in metadata}

#     for reg_no, name in unique_students:
#         status = "Present" if reg_no in present_reg else "Absent"
#         all_data.append([reg_no, name, status, timestamp])

#     df = pd.DataFrame(all_data, columns=["reg_no", "name", "Status", "Timestamp"])
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df.to_csv(output_csv, index=False)
#     print(f"[✅] Attendance marked in {output_csv}")

import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime

def mark_attendance(group_photo_path):
    # Load encodings (tuple: known_encodings, known_names)
    with open("output/encodings.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)

    # Load group photo
    image = face_recognition.load_image_file(group_photo_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    print(f"[ℹ️] Total faces detected in group photo: {len(face_encodings)}")

    present_students = set()

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if True in matches:
            best_match_index = face_distances.argmin()
            name = known_names[best_match_index]
            print(f"[✅] Match found: {name}")
            present_students.add(name)
        else:
            print("[❌] No match for a face in the group photo.")

    all_students = set(known_names)
    absent_students = all_students - present_students

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("output/attendance.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Reg. No.", "Name", "Status", "Timestamp"])

        for student in present_students:
            reg_no, name = student.split("_", 1)
            writer.writerow([reg_no, name, "Present", now])

        for student in absent_students:
            reg_no, name = student.split("_", 1)
            writer.writerow([reg_no, name, "Absent", now])

    print("[✅] Attendance marked in output/attendance.csv")

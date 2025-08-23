# import face_recognition
# import os
# import pickle

# def generate_encodings(faces_dir, output_file='output/encodings.pkl'):
#     known_encodings = []
#     known_metadata = []

#     for folder in os.listdir(faces_dir):
#         reg_no, name = folder.split("_", 1)
#         folder_path = os.path.join(faces_dir, folder)

#         for image_file in os.listdir(folder_path):
#             img_path = os.path.join(folder_path, image_file)
#             image = face_recognition.load_image_file(img_path)
#             encodings = face_recognition.face_encodings(image)

#             if encodings:
#                 known_encodings.append(encodings[0])
#                 known_metadata.append({
#                     'reg_no': reg_no,
#                     'name': name,
#                     'image': image_file
#                 })

#     with open(output_file, 'wb') as f:
#         pickle.dump((known_encodings, known_metadata), f)
#     print(f"[✅] Encodings saved to {output_file}")




import face_recognition
import os
import cv2
import pickle

def generate_encodings(faces_dir):
    known_encodings = []
    known_names = []

    for folder in os.listdir(faces_dir):
        folder_path = os.path.join(faces_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        try:
            reg_no, name = folder.split("_", 1)
        except ValueError:
            print(f"[⚠️] Skipping folder '{folder}' — name format must be <reg_no>_<name>")
            continue

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # Skip non-image files and directories
            if not os.path.isfile(image_path) or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"[⚠️] Skipping invalid image: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"[❌] Could not read image: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(f"{reg_no}_{name}")

    # Save encodings
    data = (known_encodings, known_names)

    os.makedirs("output", exist_ok=True)
    with open("output/encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    print("[✅] Encodings saved to output/encodings.pkl")

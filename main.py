from utils.image_capture import capture_images
from utils.face_encoder import generate_encodings
from utils.face_detector import mark_attendance
import os

if __name__ == "__main__":
    print("1. Capture student images")
    print("2. Generate face encodings")
    print("3. Mark attendance from group photo")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        reg_no = input("Enter registration number: ").strip()
        name = input("Enter student name: ").strip()

        # Folder name must be: reg_no_name (e.g., 22157147046_Ramesh Kumar)
        folder_name = f"{reg_no}_{name}"
        save_path = os.path.join("data/faces", folder_name)

        capture_images(save_path, reg_no, name)

    elif choice == '2':
        generate_encodings("data/faces")

    elif choice == '3':
        photo_name = input("Enter group photo filename (inside data/group_photos/): ").strip()
        photo_path = f"data/group_photos/{photo_name}"
        mark_attendance(photo_path)

    else:
        print("[❌] Invalid choice.")

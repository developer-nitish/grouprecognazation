import cv2
import os

def capture_images(reg_no, name, save_dir="data/faces", num_images=20):
    person_dir = os.path.join(save_dir, reg_no)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"[ℹ️] Starting image capture for {name} ({reg_no})...")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[❌] Failed to capture frame.")
            break

        img_path = os.path.join(person_dir, f"{count+1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[✅] Saved: {img_path}")
        count += 1

        # Show the frame
        cv2.imshow("Capturing Images - Press 'q' to exit early", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[ℹ️] Image capture completed.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reg_no = input("Enter Registration Number: ")
    name = input("Enter Name: ")
    capture_images(reg_no, name)

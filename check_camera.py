import cv2

print("Checking available cameras...")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i} is available and working")
            print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            print(f"✗ Camera {i} opened but cannot read frames")
        cap.release()
    else:
        print(f"✗ Camera {i} is not available")

print("\nIf no cameras are available, please check:")
print("1. System Settings → Privacy & Security → Camera")
print("2. Enable camera access for Terminal/Python")
print("3. Restart your terminal after granting permissions")

import os
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

import pyzed.sl as sl
import numpy as np
import cv2

def main():
    cam = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1200
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera:", status)
        return

    info = cam.get_camera_information()
    print("Model   :", info.camera_model)
    print("Serial  :", info.serial_number)
    print("Press Q to quit.")

    rgb_mat   = sl.Mat()
    depth_mat = sl.Mat()
    runtime   = sl.RuntimeParameters()

    while True:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        
        cam.retrieve_image(rgb_mat, sl.VIEW.LEFT)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        rgb   = rgb_mat.get_data()[:, :, :3]          # drop alpha, keep BGR
        depth = depth_mat.get_data()                   # float32 metres

        # Normalise depth to 8-bit for display (clip 0–10 m)
        depth_vis = np.clip(depth, 0, 10) / 10.0
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Stack side-by-side
        h = min(rgb.shape[0], depth_vis.shape[0])
        combined = np.hstack([rgb[:h], depth_vis[:h]])

        cv2.imshow("ZED-X  |  RGB (left)  +  Depth", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

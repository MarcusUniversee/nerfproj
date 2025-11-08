import cv2
import os
import numpy as np

TAG_SIZE = 0.06  # 60 mm
relative_corners = np.array([
    [0.0,       0.0,        0.0],
    [TAG_SIZE,  0.0,        0.0],
    [TAG_SIZE,  TAG_SIZE,   0.0],
    [0.0,       TAG_SIZE,   0.0]
], dtype=np.float32)

tag_origins = np.array([
    [0.00, 0.00000, 0.0],
    [0.09, 0.00000, 0.0],
    [0.00, 0.07567, 0.0],
    [0.09, 0.07567, 0.0],
    [0.00, 0.15134, 0.0],
    [0.09, 0.15134, 0.0],
], dtype=np.float32)

# Per-tag object points (4x3 each)
OBJ_PER_TAG = tag_origins[:, None, :] + relative_corners[None, :, :]

def get_corners(image):
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Detect ArUco markers in an image
    # Returns: corners (list of numpy arrays), ids (numpy array)
    corners, ids, _ = detector.detectMarkers(image)
    found = [None] * 6
    # Check if any markers were detected
    if ids is not None:
        # Process the detected corners
        # corners: list of length N (number of detected tags)
        #   - each element is a numpy array of shape (1, 4, 2) containing the 4 corner coordinates (x, y)
        # ids: numpy array of shape (N, 1) containing the tag IDs for each detected marker
        # Example: if 3 tags detected, corners will be a list of 3 arrays, ids will be shape (3, 1)
        image_d = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(len(ids)):
            id = int(ids[i, 0])
            if 0 <= id < 6:
                found[id] = corners[i].reshape(4, 2).astype(np.float32)
        for i, pts in enumerate(corners):
            pts = pts.reshape(4, 2).astype(int)
            cv2.polylines(image_d, [pts], isClosed=True, color=(0, 255, 0), thickness=5)
            
            # draw the ID number in red (or any color you want)
            x, y = pts[0]
            cv2.putText(image_d, str(int(ids[i, 0])), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("images with points", cv2.resize(image_d, (1280, 700)))
        cv2.waitKey(1)
        
        return found
    else:
        # No tags detected in this image, skip it
        return found

def calibrate(images):
    objectPoints = []
    imagePoints  = []
    for idx, im in enumerate(images):
        corners = get_corners(im)
        print(f"progress: {idx/50 + 0.05}")
        img_pts = []
        obj_pts = []
        for tag_id, pix in enumerate(corners):
            if pix is None:
                continue
            img_pts.append(pix)
            obj_pts.append(OBJ_PER_TAG[tag_id])
        if len(img_pts) == 0:
            continue
        img_pts = np.concatenate(img_pts, axis=0).reshape(-1, 1, 2).astype(np.float32)
        obj_pts = np.concatenate(obj_pts, axis=0).reshape(-1, 1, 3).astype(np.float32)

        imagePoints.append(img_pts)
        objectPoints.append(obj_pts)
    h, w = images[0].shape[:2]
    image_size = (w, h)
    print(f"progress: {0.90}")
    retval, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, image_size, None, None
    )
    print(f"progress: {0.95}")
    return retval, K, dist, rvecs, tvecs


if __name__ == "__main__":
    folder = "./calibration_imgs"
    images = []
    print(f"progress: {0}")
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)
            images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            count += 1
        if count == 1:
            break
    print(f"progress: {0.04}")
    retval, K, dist, rvecs, tvecs = calibrate(images)
    h, w = images[0].shape[:2]
    print("image size:", w, h)
    
    # with open("output.txt", "w") as file:
    #     # Write the string representation of the variable to the file
    #     file.write(f"Error= {retval}")
    #     file.write(f"K= {K}")
    #     file.write(f"distortion= {dist}")
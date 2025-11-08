import viser
import numpy as np
import cv2
import os
import time

#aruco marker
TAG_SIZE = 0.1 # 100 mm
world_corners = np.array([
    [0.0,       0.0,        0.0],
    [TAG_SIZE,  0.0,        0.0],
    [TAG_SIZE,  TAG_SIZE,   0.0],
    [0.0,       TAG_SIZE,   0.0]
], dtype=np.float32)

#camera intrisics gained from calibrate.py
K = np.array([
    [3.98897118e+03, 0.00000000e+00, 2.85629228e+03],
    [0.00000000e+00, 4.02275067e+03, 2.15719743e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float64)
DIST = np.array([[ 0.20170125, -0.42258863, -0.00239507, -0.00090458, -0.63641613]],
                      dtype=np.float64)

def get_corner(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is not None and len(ids) == 1:
        return corners[0].reshape(4, 2).astype(np.float32)  
    else:
        return None

def estimator(images):
    extrinsic = []
    camera_to_world = []
    succ_images_ids = []
    for i, im_c in enumerate(images):
        im = cv2.cvtColor(im_c, cv2.COLOR_BGRA2GRAY)
        corners = get_corner(im)
        if corners is not None and len(corners) == 4:
            success, rvec, tvec = cv2.solvePnP(world_corners, corners, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                R, _ = cv2.Rodrigues(rvec)   # Convert rod. vector → rotation matrix (3x3)
                t = tvec.reshape(3,1)        # Ensure column shape (3x1)

                # Build the 3x4 extrinsic matrix [R | t]
                ex = np.hstack((R, t))
                extrinsic.append(ex)
                R_wc = R.T
                t_wc = -R_wc @ t

                c2w = np.eye(4, dtype=np.float64)
                c2w[:3, :3] = R_wc
                c2w[:3, 3]  = t_wc[:, 0]
                camera_to_world.append(c2w)
                print(f"progress: {0.16 + i/100} Success")
                succ_images_ids.append(i)
                continue
        #extrinsic.append(None)
        #camera_to_world.append(None)
        print(f"progress: {0.16 + i/100} None")
    return extrinsic, camera_to_world, succ_images_ids

def visualize(images, c2w_list):

    server = viser.ViserServer(share=True)
    H, W = images[0].shape[:2]
    # Example of visualizing a camera frustum (in practice loop over all images)
    # c2w is the camera-to-world transformation matrix (3x4), and K is the camera intrinsic matrix (3x3)
    for i in range(len(images)):
        c2w = c2w_list[i]
        if c2w is None:
            continue
        img = images[i]
        server.scene.add_camera_frustum(
            f"/cameras/{i}", # give it a name
            fov=2 * np.arctan2(H / 2, K[0, 0]), # field of view
            aspect=W / H, # aspect ratio
            scale=0.02, # scale of the camera frustum change if too small/big
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, # orientation in quaternion format
            position=c2w[:3, 3], # position of the camera
            image=img # image to visualize
        )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run

def undistort(images):
    h, w = images[0].shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (w,h), 1, (w,h))
    undistorted_images = []
    print(f"progress: {0.52}")
    for i, img in enumerate(images):
        undistorted = cv2.undistort(img, K, DIST, None, newcameramtx)
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        cv2.imshow("undistorted", cv2.resize(undistorted, (1280, 700)))
        cv2.waitKey(1)
        undistorted_images.append(undistorted)
        print(f"progress: {0.53 + i/100}")
    adjK = newcameramtx.copy()
    adjK[0, 2] -= x
    adjK[1, 2] -= y
    return undistorted_images, adjK, roi


if __name__ == "__main__":
    folder = "./object_imgs"
    images = []
    images_bgr = []
    print(f"progress: {0}")
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            images.append(img)
            images_bgr.append(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
    print(f"progress: {0.15}")
    extrinsic_cam, c2w, succ_ids = estimator(images)
    images_bgr = [images_bgr[i] for i in succ_ids]
    print(f"progress: {0.50}")
    undistorted, adjK, roi = undistort(images_bgr)
    undistorted = np.array(undistorted)
    N_test = int(len(undistorted) * 0.9)
    N_train = int(N_test * 0.9)
    print(undistorted.shape)
    c2w = np.asarray(c2w, dtype=np.float64)
    H, W = images[0].shape[:2]

    np.savez(
        'headphone_data.npz',
        images_train=undistorted[:N_train, :, :, :],    # (N_train, H, W, 3)
        c2ws_train=c2w[:N_train, :, :],        # (N_train, 4, 4)
        images_val=undistorted[N_train:N_test, :, :, :],        # (N_val, H, W, 3)
        c2ws_val=c2w[N_train:N_test, :, :],            # (N_val, 4, 4)
        c2ws_test=c2w[N_test:, :, :],          # (N_test, 4, 4)
        focal = 2 * np.arctan2(W / 2, K[0, 0])                 # float
    )
    #visualize(images, c2w)
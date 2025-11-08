import numpy as np

#transforms from camera to world coordinates
def transform(c2w, x_c):
    #x_c is (N, 3)
    if x_c.shape[1] == 3:
        ones = np.ones((x_c.shape[0], 1), dtype=x_c.dtype)
        x_c_h = np.hstack([x_c, ones]) #homgenous coordinates
    else:
        x_c_h = x_c
    #convert x_c to (4, N) to get: 4xN. then take transpose to get Nx4
    x_w_h = (c2w @ x_c_h.T).T
    return x_w_h[:, :3] / x_w_h[:, 3] #normalize to (N, 3)

def pixel_to_camera(K, uv, s):
    #s * K inv @ uv = x_c
    #uv is (N, 2)
    if uv.shape[1] == 2:
        ones = np.ones((uv.shape[0], 1), dtype=uv.dtype)
        uv_h = np.hstack([uv, ones]) #homgenous coordinates
    else:
        uv_h = uv
    #convert uv to (3, N) to get 3xN. then take transpose to get Nx3
    x_c = s * (np.linalg.inv(K) @ uv_h.T).T
    return x_c

def pixel_to_ray(K, c2w, uv):
    # get first three rows, last column
    r_0 = c2w[:3, 3][None, :] #(1, 3)
    x_w = transform(c2w, pixel_to_camera(K, uv, 1)) # s = 1 from spec
    r_d = (x_w - r_0) / np.sqrt((x_w[:, 0] - r_0[:, 0]) ** 2 + (x_w[:, 1] - r_0[:, 1]) ** 2 + (x_w[:, 2] - r_0[:, 2]) ** 2)
    return r_0, r_d
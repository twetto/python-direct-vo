import numpy as np
import cv2

K = np.eye(3)
D = np.zeros(4)
res = (752, 480)
R = np.eye(3)
T = np.array([0.11, 0, 0]) # shape (3,)

try:
    cv2.stereoRectify(K, D, K, D, res, R, T, alpha=0.0)
    print("Success with shape (3,)")
except Exception as e:
    print(f"Failed with shape (3,): {e}")

T2 = T.reshape(3, 1)
try:
    cv2.stereoRectify(K, D, K, D, res, R, T2, alpha=0.0)
    print("Success with shape (3, 1)")
except Exception as e:
    print(f"Failed with shape (3, 1): {e}")

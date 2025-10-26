import cv2
import numpy as np


def fb_blur_fusion_foreground_estimator_1(image, alpha, r=90):
    alpha = alpha[:, :, None]
    return fb_blur_fusion_foreground_estimator(image, f=image, b=image, alpha=alpha, r=r)[0]


def fb_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    alpha = alpha[:, :, None]
    f, blur_b = fb_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return fb_blur_fusion_foreground_estimator(image, f, blur_b, alpha, r=6)[0]


def fb_blur_fusion_foreground_estimator(image, f, b, alpha, r=90):
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_fa = cv2.blur(f * alpha, (r, r))
    blurred_f = blurred_fa / (blurred_alpha + 1e-5)

    blurred_b1_a = cv2.blur(b * (1 - alpha), (r, r))
    blurred_b = blurred_b1_a / ((1 - blurred_alpha) + 1e-5)
    f = blurred_f + alpha * (image - alpha * blurred_f - (1 - alpha) * blurred_b)
    f = np.clip(f, 0, 1)
    return f, blurred_b
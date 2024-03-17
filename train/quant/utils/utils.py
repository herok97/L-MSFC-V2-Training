import cv2

def letterbox(
    img, height=608, width=1088, color=(127.5, 127.5, 127.5)
):  
    shape = img.shape[:2]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )
    dw = (width - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, dw, dh, shape
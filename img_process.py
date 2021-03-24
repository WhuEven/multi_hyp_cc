import cv2.cv2 as cv2
import numpy as np
import os

def diag_pro(img, diag):
    pixels_converted = np.matmul(img, np.diag(diag))
    return pixels_converted
def apply_whitebalance(im, illum):
    for channel in range(im.shape[2]):
        im[:, :, channel] /= max(illum[channel], 0.00001)

    return np.clip(im, 0.0, 1.0)

def transfer_16bit_to_8bit(image_path):
    
    # min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    # 或者下面一种写法
    # image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    one_zero = np.array((image_16bit) / (max_16bit))
    # image_8bit = np.array(np.rint(255 * (image_16bit / 25535)), dtype=np.uint8)
    # print(image_16bit.dtype)
    # print('16bit dynamic range: %d - %d' % (min_16bit, max_16bit))
    # print(image_8bit.dtype)
    # print('8bit dynamic range: %d - %d' % (np.min(image_8bit), np.max(image_8bit)))
    process_img = apply_whitebalance(one_zero,diag)
    image_8bit = np.array(255 * process_img,dtype=np.uint8)
    # image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
    # print(np.max(image_8bit))
    # print(np.min(image_8bit))
    cv2.namedWindow("ori", 0)
    cv2.resizeWindow("ori", 640, 480)
    cv2.imshow("ori",image_8bit)
    cv2.waitKey()

image_path = './eryixing/Canon1DsMkIII_0064.PNG'
image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
diag = [0.4942323,0.70926505,0.5026704]
# diag = [0.5309567, 0.7829728, 0.32409668]
# process_img = diag_pro(image_16bit,diag)
# process_img = apply_whitebalance(image_16bit,diag)

transfer_16bit_to_8bit(image_16bit)

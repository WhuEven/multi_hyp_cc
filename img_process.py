import cv2.cv2 as cv2
import numpy as np
import os

def apply_ccm(im, ccm):
    pixels = np.reshape(im, [im.shape[0]*im.shape[1], im.shape[2]])
    pixels_converted = np.matmul(pixels, ccm.T)
    im = np.reshape(pixels_converted, im.shape)
    return im

def blacklevel_saturation_correct(im, sensor, saturation_scale = 1.0):
    original_dtype = im.dtype
    max_value = np.iinfo(original_dtype).max

    im = im.astype(np.float)
    im = im - sensor["black_level"]
    im[im < 0] = 0
    saturation = sensor["saturation"] - sensor["black_level"]
    im = im / (saturation_scale * saturation)
    im[im > 1] = 1.0
    im = (max_value*im)
    im = im.astype(original_dtype)

    return im

def apply_whitebalance(im, illum):
    for channel in range(im.shape[2]):
        im[:, :, channel] /= max(illum[channel], 0.00001)

    return np.clip(im, 0.0, 1.0)

def convert_for_display(im_src, illum, ccm):
    im = np.copy(im_src)
    if illum is not None:
        im = apply_whitebalance(im, illum)
    im = apply_ccm(im, ccm)
    im = apply_gamma_correction(im)
    return im

def apply_gamma_correction(im):
    a = 0.055
    b = 1.055
    t = 0.0031308
    im[im < 0] = 0
    im[im > 1] = 1
    #im = (im_linear * 12.92) .* (im_linear <= t) ...
    #   + ((1 + a)*im_linear.^(1/2.4) - a) .* (im_linear > t);
    im = np.multiply(im * 12.92, im < t) + np.multiply(b*np.power(im, (1/2.4)) - a, im >= t)
    im[im < 0] = 0
    im[im > 1] = 1
    return im

def transfer_16bit_to_8bit(im_i):
    
    # min_16bit = np.min(image_16bit)
    # max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    # 或者下面一种写法
    # image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    
    # one_zero = np.array((image_16bit) / (max_16bit))
    # image_8bit = np.array(np.rint(255 * (image_16bit / 25535)), dtype=np.uint8)
    # print(image_16bit.dtype)
    # print('16bit dynamic range: %d - %d' % (min_16bit, max_16bit))
    # print(image_8bit.dtype)
    # print('8bit dynamic range: %d - %d' % (np.min(image_8bit), np.max(image_8bit)))
    
    # process_img = apply_whitebalance(process_img,diag)
    
    # image_8bit = np.array(255 * process_img,dtype=np.uint8)
    # image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
    # print(np.max(image_8bit))
    # print(np.min(image_8bit))
    # im_i = image_16bit - min_16bit


    im_i = blacklevel_saturation_correct(im_i, sensor_i, saturation_scale = 0.95)
    im_i = im_i.astype(np.float) / np.iinfo(im_i.dtype).max
    im_i = convert_for_display(im_i, illuminant, np.array(ccm))
    im_i = (255*im_i).astype(np.uint8)
    # im_i = cv2.cvtColor(im_i, cv2.COLOR_RGB2BGR)


    cv2.namedWindow("ori", 0)
    cv2.resizeWindow("ori", 640, 480)
    cv2.imshow("ori",im_i)
    cv2.waitKey()

image_path = './eryixing/Canon1DsMkIII_0064.PNG'
image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
ccm = [[1.7247124727955703e+00,  -7.7909567189851892e-01,   5.4383199102948671e-02],
[-1.4362260963839543e-01,   1.4631606310333440e+00,  -3.1953802139494858e-01],
 [5.8852418690761139e-02,  -4.6253019878880824e-01,   1.4036777800980471e+00]]
# illuminant = [0.4942323,0.70926505,0.5026704]#gt
# illuminant = [0.5309567, 0.7829728, 0.32409668]
illuminant = [1,1,1]
# process_img = diag_pro(image_16bit,diag)
# process_img = apply_whitebalance(image_16bit,diag)

sensor_i = {"black_level":1024, "saturation":15279}

transfer_16bit_to_8bit(image_16bit)

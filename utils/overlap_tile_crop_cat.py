from PIL import Image
import numpy as np
import math


def symmetric_pad_img(origin_img, pad_pixel=100):
    origin_img_array = np.array(origin_img)
    padded_img_array = np.pad(origin_img_array,
                              pad_width=((pad_pixel, pad_pixel), (pad_pixel, pad_pixel), (0, 0)),
                              mode='symmetric')
    padded_img = Image.fromarray(padded_img_array.astype('uint8')).convert('RGB')
    return padded_img


def cropfullimg(origin_img, patch_size=512, pad_pixel=100):
    center_patch_size = patch_size - 2 * pad_pixel
    (w, h) = origin_img.size
    p_w = math.ceil(w / center_patch_size)
    p_h = math.ceil(h / center_patch_size)

    origin_img_array = np.array(origin_img)
    padded_img_array = np.pad(origin_img_array,
                              pad_width=((pad_pixel, pad_pixel), (pad_pixel, pad_pixel), (0, 0)),
                              mode='symmetric')
    padded_img = Image.fromarray(padded_img_array)
    (padded_w, padded_h) = padded_img.size

    img_patchs = []
    center_crops = []
    big_patch_crops = []
    for h_i in range(p_h):
        for w_i in range(p_w):
            if(w_i == (p_w - 1) and h_i != (p_h - 1)):
                center_crops.append((w - center_patch_size, h_i * center_patch_size,
                                     w, center_patch_size * (h_i + 1)))
                big_patch_crops.append((padded_w - patch_size, h_i * center_patch_size,
                                        padded_w, center_patch_size * h_i + patch_size))

            elif(h_i == (p_h - 1) and w_i != (p_w - 1)):
                center_crops.append((w_i * center_patch_size, h - center_patch_size,
                                     center_patch_size * (w_i + 1), h))
                big_patch_crops.append((w_i * center_patch_size, padded_h - patch_size,
                                        center_patch_size * w_i + patch_size, padded_h))

            elif(w_i == (p_w - 1) and h_i == (p_h - 1)):
                center_crops.append((w - center_patch_size, h - center_patch_size, w, h))
                big_patch_crops.append((padded_w - patch_size, padded_h - patch_size, padded_w, padded_h))
            else:
                center_crops.append((w_i * center_patch_size, h_i * center_patch_size,
                                     center_patch_size * (w_i + 1), center_patch_size * (h_i + 1)))
                big_patch_crops.append((w_i * center_patch_size, h_i * center_patch_size,
                                        center_patch_size * (w_i + 1) + 2 * pad_pixel,
                                        center_patch_size * (h_i + 1) + 2 * pad_pixel))

            img_patch = padded_img_array[big_patch_crops[h_i * p_w + w_i][1]:big_patch_crops[h_i * p_w + w_i][3],
                        big_patch_crops[h_i * p_w + w_i][0]:big_patch_crops[h_i * p_w + w_i][2]] # high range, wide range
            img_patch = Image.fromarray(img_patch)
            img_patchs.append(img_patch)

    return img_patchs, center_crops, big_patch_crops


def concat_img_patchs(img_patchs, crops, full_img_size, patch_size=512, pad_pixel=100):
    center_patch_size = patch_size - 2 * pad_pixel
    (w, h) = full_img_size
    p_w = math.ceil(w / center_patch_size)
    p_h = math.ceil(h / center_patch_size)

    crops = crops
    concat_img = Image.new('RGB', (w, h), (0, 0, 0))
    for h_i in range(p_h):
        for w_i in range(p_w):
            img_patch = img_patchs[h_i * p_w + w_i]
            center_img_patch = img_patch.crop((pad_pixel, pad_pixel,
                                               patch_size - pad_pixel, patch_size - pad_pixel))
            concat_img.paste(center_img_patch, crops[h_i * p_w + w_i])

    return concat_img


if __name__ == '__main__':
    origin_img = Image.open('G:/dataset/temp/1100-14-300-h (4)_crop.tif')
    padded_img = symmetric_pad_img(origin_img)
    padded_img.save('G:/dataset/temp/1100-14-300-h (4)_crop_pad.png')
    # img_patchs, center_crops, big_patch_crops = cropfullimg(origin_img)
    # concat_img = concat_img_patchs(img_patchs, center_crops, (1023, 767))
    # concat_img.show()
    # for patch in img_patchs:
    #     patch.show()
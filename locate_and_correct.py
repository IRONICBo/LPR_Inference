# -*- coding:utf-8 -*-
import cv2
import numpy as np

def locate_and_correct(img_src, img_mask):
    """
    :param img_src: origin image
    :param img_mask: binary image
    :return: corrected image
    """
    img_mask[:, :, 0] = img_mask[:, :, 0].astype(np.uint8)
    try:
        contours, hierarchy = cv2.findContours(img_mask[:, :, 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret, contours, hierarchy = cv2.findContours(img_mask[:, :, 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return [], []
    else:
        Lic_img = []
        bbox = []
        for ii, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)
            img_cut_mask = img_mask[y:y + h, x:x + w]
            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect).astype(np.int32)
                cont = cont.reshape(-1, 2).tolist()
                box = sorted(box, key=lambda xy: xy[0])
                box_left, box_right = box[:2], box[2:]
                box_left = sorted(box_left, key=lambda x: x[1])
                box_right = sorted(box_right, key=lambda x: x[1])
                box = np.array(box_left + box_right)
                x0, y0 = box[0][0], box[0][1]
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]

                # correct the point to distance
                def point_to_line_distance(X, Y):
                    if x2 - x0:
                        k_up = (y2 - y0) / (x2 - x0)
                        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                    else:
                        d_up = abs(X - x2)
                    if x1 - x3:
                        k_down = (y1 - y3) / (x1 - x3)
                        d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                    else:
                        d_down = abs(X - x1)
                    return d_up, d_down

                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
                for each in cont:
                    x, y = each[0], each[1]
                    dis0 = (x - x0) ** 2 + (y - y0) ** 2
                    dis1 = (x - x1) ** 2 + (y - y1) ** 2
                    dis2 = (x - x2) ** 2 + (y - y2) ** 2
                    dis3 = (x - x3) ** 2 + (y - y3) ** 2
                    d_up, d_down = point_to_line_distance(x, y)
                    weight = 0.975
                    if weight * d_up + (1 - weight) * dis0 < d0:
                        d0 = weight * d_up + (1 - weight) * dis0
                        l0 = (x, y)
                    if weight * d_down + (1 - weight) * dis1 < d1:
                        d1 = weight * d_down + (1 - weight) * dis1
                        l1 = (x, y)
                    if weight * d_up + (1 - weight) * dis2 < d2:
                        d2 = weight * d_up + (1 - weight) * dis2
                        l2 = (x, y)
                    if weight * d_down + (1 - weight) * dis3 < d3:
                        d3 = weight * d_down + (1 - weight) * dis3
                        l3 = (x, y)


                bbox = np.array([l0[0], l0[1], l3[0], l3[1]])
                p0 = np.float32([l0, l1, l2, l3])
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])
                transform_mat = cv2.getPerspectiveTransform(p0, p1)
                lic = cv2.warpPerspective(img_src, transform_mat, (240, 80))
                Lic_img.append(lic)
    return Lic_img, bbox
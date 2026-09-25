import cv2
import numpy as np
from matplotlib import pyplot as plt

def Harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    reference_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite('solutions/HCD_image.png', reference_image)

    cv2.imshow('dst', reference_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def SIFT(image_to_align, reference_image, max_features, good_match_precent):
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray_align.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        gray_ref = cv2.polylines(gray_ref, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Aligning av bildet for utskriving
        align_h, align_w = gray_ref.shape
        align = cv2.warpPerspective(image_to_align, M, (align_w, align_h))

        cv2.imshow('align', align)
        cv2.imwrite('solutions/SIFT_aligned_image.png', align)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatches(gray_align, kp1, gray_ref, kp2, good, None, **draw_params)

    cv2.imwrite('solutions/SIFT_matches_image.png', img3)
    plt.imshow(img3, 'gray'), plt.show()



def main():
    image = cv2.imread('reference_img.png')
    aligning = cv2.imread('align_this.jpg')

    Harris_corner_detection(image)

    SIFT(aligning, image, 10, 0.7)


if __name__ == '__main__':
    main()
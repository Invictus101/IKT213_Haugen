import cv2
import numpy as np

def sobel_edge_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (3, 3),0, borderType=cv2.BORDER_DEFAULT)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)
    sobelxy = cv2.magnitude(sobelx, sobely)

    normalized = cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Sobel Edges', normalized)
    cv2.imwrite('solutions/sobel_edges_lambo.png', normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred = cv2.GaussianBlur(image, (3, 3), 0, borderType=cv2.BORDER_DEFAULT)

    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    cv2.imshow('Canny Edges', edges)
    cv2.imwrite('solutions/canny_edges_lambo.png', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(image_gray,template_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imshow('Template match', image)
    cv2.imwrite('solutions/template_match.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, scale_factor: int, up_or_down: str):
    rows, cols, _channels = map(int, image.shape)

    if up_or_down == 'up':
        image = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
        cv2.imshow('Upsized image', image)
        cv2.imwrite('solutions/upsized_lambo.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif up_or_down == 'down':
        image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
        cv2.imshow('Downsized image', image)
        cv2.imwrite('solutions/downsized_lambo.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    image_lambo = cv2.imread('lambo.png')
    image_shapes = cv2.imread('shapes-1.png')
    image_shapes_template = cv2.imread('shapes_template.jpg')

    print("Choose a Task:")
    print("1. Sobel edge detection")
    print("2. Canny edge detection")
    print("3. Template matching")
    print("4. Resize")
    print("5. All")


    choice = int(input("Enter choice: "))
    match choice:
        case 1:
            sobel_edge_detection(image_lambo)

        case 2:
            canny_edge_detection(image_lambo, 50, 50)

        case 3:
            template_match(image_shapes, image_shapes_template)

        case 4:
            resize(image_lambo, 2, "up")
            resize(image_lambo, 2, "down")

        case 5:
            sobel_edge_detection(image_lambo)

            canny_edge_detection(image_lambo, 50, 50)

            template_match(image_shapes, image_shapes_template)

            resize(image_lambo, 2, "up")
            resize(image_lambo, 2, "down")

        case _:
            print("Invalid choice")


if __name__ == '__main__':
    main()
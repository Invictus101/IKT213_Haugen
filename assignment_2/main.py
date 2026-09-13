import cv2
import numpy as np

def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imshow("Padding_iris-1.png", reflect)
    cv2.imwrite("solutions/Padding_iris-1.png", reflect)


def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imshow("Cropped_iris-1.png", cropped)
    cv2.imwrite("solutions/Cropped_iris-1.png", cropped)

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    cv2.imshow("Resized_iris-1.png", resized)
    cv2.imwrite("solutions/Resized_iris-1.png", resized)

def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]

    cv2.imshow("Copied_iris-1.png", emptyPictureArray)
    cv2.imwrite("solutions/Copied_iris-1.png", emptyPictureArray)

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale_iris-1.png", gray)
    cv2.imwrite("solutions/Grayscale_iris-1.png", gray)

def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Hsv_iris-1.png", hsv)
    cv2.imwrite("solutions/Hsv_iris-1.png", hsv)

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):

                new_value = int(image[y, x, c]) + hue

                #Holde innen color value
                if new_value > 255:
                    new_value = 255

                if new_value < 0:
                    new_value = 0

                emptyPictureArray[y, x, c] = new_value

    cv2.imshow("Hue_shifted_iris-1.png", emptyPictureArray)
    cv2.imwrite("solutions/Hue_shifted_iris-1.png", emptyPictureArray)

def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15),0, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow("Smoothed_iris-1.png", blurred)
    cv2.imwrite("solutions/Smoothed_iris-1.png", blurred)

def rotation(image, rotation_angle):

    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("90_degrees_iris-1.png", rotated)
        cv2.imwrite("solutions/90_degrees_iris-1.png", rotated)


    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow("180_degrees_iris-1.png", rotated)
        cv2.imwrite("solutions/180_degrees_iris-1.png", rotated)



def main():
    image = cv2.imread('iris-1.png')
    height, width, channels = image.shape

    print("Choose an operation:")
    print("1. Padding")
    print("2. Cropping")
    print("3. Resize")
    print("4. Manual copy")
    print("5. Grayscale")
    print("6. HSV")
    print("7. Color shifting")
    print("8. Smoothing")
    print("9. Rotation")
    print("10. Run all")

    choice = int(input("Enter choice: "))
    match choice:

        case 1:
            padding(image, 100)

        case 2:
            crop(image,200, width - 130,200, height - 130)

        case 3:
            resize(image, 200, 200)

        case 4:
            emptyPicture = np.zeros((height, width, 3), dtype=np.uint8)
            copy(image, emptyPicture)

        case 5:
            grayscale(image)

        case 6:
            hsv(image)

        case 7:
            emptyHuePicture = np.zeros((height, width, 3), dtype=np.uint8)
            hue_shifted(image, emptyHuePicture, 50)

        case 8:
            smoothing(image)

        case 9:
            rotation(image, 90)
            rotation(image, 180)

        case 10:
            # 1. Padding
            padding(image, 100)

            # 2. Cropping
            crop(image,200, width - 130,200, height - 130)

            # 3. Resize
            resize(image, 200, 200)

            # 4. Manual copy
            emptyPicture = np.zeros((height, width, 3),dtype=np.uint8)
            copy(image, emptyPicture)

            # 5. Grayscale
            grayscale(image)

            # 6. HSV
            hsv(image)

            # 7. Color shifting
            emptyHuePicture = np.zeros((height, width, 3), dtype=np.uint8)
            hue_shifted(image, emptyHuePicture, 50)

            # 8. Smoothing
            smoothing(image)

            # 9. Rotation
            rotation(image, 90)
            rotation(image, 180)


        case _:
            print("Invalid choice")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

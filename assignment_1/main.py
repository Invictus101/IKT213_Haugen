import numpy as np
import cv2

def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_image_information(image):
    height, width, channels = image.shape

    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("data type:", image.dtype)

def save_camera_information():
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    while (True):
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")

    cap.release()
    cv2.destroyAllWindows()

def main():
    image = cv2.imread('iris-1.jpg')
    # Image working test
    # show_image(image)
    #Ass4
    print_image_information(image)
    # #Ass5
    save_camera_information()

if __name__ == '__main__':
    main()

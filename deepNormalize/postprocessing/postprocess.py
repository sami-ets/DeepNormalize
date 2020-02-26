import cv2


if __name__ == '__main__':

    file = "iseg_input.png"

    image = cv2.imread(file, 0)

    image[image > 240] = 0

    cv2.imwrite("postprocessed_iseg_input.png", image)
    print("Hello")
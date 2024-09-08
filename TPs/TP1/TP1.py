import cv2
import numpy as np
import math


fragment_directory = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1/frag_eroded/"
fragment_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1/fragments.txt"
final_image_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1/Michelangelo_ThecreationofAdam_1707x775.jpg"


def load_fragments(fragment_path):
    # Initialize a list to store the data
    fragments = []

    # Open the file and read line by line
    with open(fragment_path, 'r') as file:
        for line in file:
            # Split each line by spaces and convert the values to appropriate types
            values = line.split()
            # Convert the first three to integers and the last one to float
            values = [int(values[0]), int(values[1]), int(values[2]), float(values[3])]
            # Append the parsed values to the data list
            fragments.append(values)

    return fragments


def load_images(fragments, images_path):
    images = []
    for i in range(len(fragments)):
        image_name = "frag_eroded_" + str(fragments[i][0]) + ".png"
        # Load an image from file
        image = cv2.imread(images_path + image_name)

        # Check if the image was loaded successfully
        if image is None:
            print("Error: Could not load image.")
        else:
            print("Image loaded successfully.")
            # Display the image
            #cv2.imshow('Loaded Image', image)

        images.append(image)

    return images


def image_preprocessing(image):
    # Convert the image to RGBA (to add an alpha channel)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # Define the mask for black pixels
    # Black pixels will have RGB values of (0, 0, 0)
    black_mask = (image_rgba[:, :, 0] == 0) & (image_rgba[:, :, 1] == 0) & (image_rgba[:, :, 2] == 0)

    # Set the alpha channel to 0 (transparent) where the mask is True
    image_rgba[black_mask, 3] = 0

    # To reshape the image, find the bounding box of the non-transparent area
    # Convert the image to grayscale and find contours
    gray_image = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_image = image_rgba

    # If the image is already cropped, skip the crop part
    if len(contours) != 0:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image using the bounding box
        cropped_image = image_rgba[y:y + h, x:x + w]

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    return cropped_image


def rotate_image(image, angle):
    # Rotate the image in another function
    height, width, _ = image.shape

    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def place_fragments(blank_image, fragments_data, fragments_images):
    x_min = y_min = 0
    y_max, x_max, _ = blank_image.shape
    print("x_max :", x_max)
    print("y_max :", y_max)

    for i in range(len(fragments_data)):
        x_i = fragments_data[i][1]
        y_i = fragments_data[i][2]
        angle = fragments_data[i][3]
        image = fragments_images[i]

        image = image_preprocessing(image)
        rotated_image = rotate_image(image, angle)

        rotated_image_2 = image_preprocessing(rotated_image)

        height, width, _ = rotated_image_2.shape


        print("======================== i : ", i)
        print("X : ", x_i)
        print("Y : ", y_i)
        print("SHAPE : ", rotated_image_2.shape)
        print("Width : ", width)
        print("height : ", height)

        x_1 = math.ceil(x_i - (width / 2))
        y_1 = math.ceil(y_i - (height / 2))
        #x_1 = max(x_1, x_min)
        #y_1 = max(y_1, y_min)
        print("x_1 : ", x_1)
        print("y_1 : ", y_1)

        x_2 = math.ceil(x_i + (width / 2))
        y_2 = math.ceil(y_i + (height / 2))
        #x_2 = min(x_2, x_max)
        #y_2 = min(y_2, y_max)
        print("x_2 : ", x_2)
        print("y_2 : ", y_2)

        blank_image[y_1:y_2, x_1:x_2] = rotated_image_2


    return blank_image


def show_image(image):
    cv2.imshow('Image', image)

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    fragments_data = load_fragments(fragment_path)
    fragments_images = load_images(fragments_data, fragment_directory)

    blank_image = cv2.imread(final_image_path)
    final_image = place_fragments(blank_image, fragments_data, fragments_images)
    show_image(final_image)


if __name__ == '__main__':
   main()


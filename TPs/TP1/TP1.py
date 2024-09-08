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

    # If the image is already cropped, skip this part
    if len(contours) != 0:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image using the bounding box
        cropped_image = image_rgba[y:y + h, x:x + w]

    # Convert the image to RGB
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    return cropped_image


def rotate_image(image, angle):
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
        # Get the data from the list
        x_i = fragments_data[i][1]
        y_i = fragments_data[i][2]
        angle = fragments_data[i][3]
        image = fragments_images[i]

        # Remove black and reshape the image
        image = image_preprocessing(image)
        # Rotate the image
        rotated_image = rotate_image(image, angle)
        # Remove the black a second time after the image rotation to reshape the image according to its new form
        final_image = image_preprocessing(rotated_image)
        # Get the height and width of the final image
        height, width, _ = final_image.shape

        print("======================== i : ", i)
        print("X : ", x_i)
        print("Y : ", y_i)
        print("SHAPE : ", final_image.shape)
        print("Width : ", width)
        print("height : ", height)

        # Compute top left coordinates of the image in the painting
        x_1 = math.ceil(x_i - (width / 2))
        y_1 = math.ceil(y_i - (height / 2))
        #x_1 = max(x_1, x_min)
        #y_1 = max(y_1, y_min)
        print("x_1 : ", x_1)
        print("y_1 : ", y_1)

        # Compute bottom right coordinates of the image in the painting
        x_2 = math.ceil(x_i + (width / 2))
        y_2 = math.ceil(y_i + (height / 2))
        #x_2 = min(x_2, x_max)
        #y_2 = min(y_2, y_max)
        print("x_2 : ", x_2)
        print("y_2 : ", y_2)

        # "Paste" the image fragment on the painting
        # Create a mask where the pixels of the image are not black
        mask = (final_image[:, :, 0] != 0) & (final_image[:, :, 1] != 0) & (final_image[:, :, 2] != 0)

        # Paste the image onto the blank_image using the mask to remove the black pixels remaining
        blank_image[y_1:y_2, x_1:x_2][mask] = final_image[mask]

        """
        Old version :
        # "Paste" the image fragment on the painting
        for i in range(width):
            for j in range(height):
                # Add the new image pixels only if they aren't black, if they aren't the contour of the image
                if (image[j, i, 0] != 0) & (image[j, i, 1] != 0) & image[j, i, 2] != 0:
                    blank_image[y_1 + j, x_1 + i] = final_image[j, i]
                    
        Initial version :
        #blank_image[y_1:y_2, x_1:x_2] = final_image
        """


    return blank_image


def show_image(image):
    # Show the image
    cv2.imshow('Image', image)
    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    # Then kill the image's window
    cv2.destroyAllWindows()


def main():
    # Load the data about the images
    fragments_data = load_fragments(fragment_path)
    # Load the images according to the data collected
    fragments_images = load_images(fragments_data, fragment_directory)
    # Get the final painting or a blank image
    blank_image = cv2.imread(final_image_path)
    # Place the fragments on the painting
    final_image = place_fragments(blank_image, fragments_data, fragments_images)
    # Show the final painting with the fragments on it
    show_image(final_image)


if __name__ == '__main__':
   main()


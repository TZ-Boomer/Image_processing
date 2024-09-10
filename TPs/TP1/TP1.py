import cv2
import numpy as np
import math


fragment_directory = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1_resources/frag_eroded/"
fragment_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1_resources/fragments.txt"
final_image_path = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP1_resources/Michelangelo_ThecreationofAdam_1707x775.jpg"


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


def rotate_image(image, angle):
    height, width, _ = image.shape

    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def place_fragments(painting, fragments_data, fragments_images):
    for i in range(len(fragments_data)):
        # Get the data from the list
        x_i = fragments_data[i][1]
        y_i = fragments_data[i][2]
        angle = fragments_data[i][3]
        fragment = fragments_images[i]

        # Rotate the image
        rotated_fragment = rotate_image(fragment, angle)
        # Get the height and width of the rotated image
        height, width, _ = rotated_fragment.shape

        # Compute top left coordinates of the image in the painting
        x_1 = math.ceil(x_i - (width / 2))
        y_1 = math.ceil(y_i - (height / 2))

        # Compute bottom right coordinates of the image in the painting
        x_2 = math.ceil(x_i + (width / 2))
        y_2 = math.ceil(y_i + (height / 2))

        # "Paste" the image fragment on the painting
        for i in range(width):
            for j in range(height):
                # Add the new image pixels only if they aren't black, if they aren't the contour of the image
                if (rotated_fragment[j, i, 0] > 3) & (rotated_fragment[j, i, 1] > 3) & (rotated_fragment[j, i, 2] > 3):
                    painting[y_1 + j, x_1 + i] = rotated_fragment[j, i]

        """      
        # Ensure the region does not go out of bounds of the blank_image
        y_2 = min(y_2, painting.shape[0])
        x_2 = min(x_2, painting.shape[1])
        y_1 = max(0, y_1)
        x_1 = max(0, x_1)

        # Adjust the rotated_image to match the valid region size
        # Sometimes, the fragment would go over the bound of the painting
        valid_height = y_2 - y_1
        valid_width = x_2 - x_1

        # Crop the image with the maximum valid size
        # DO NOT WORK IS THE FRAGMENT IS OVER THE LEFT OR THE TOP BOUND
        # The for loop works but its much slower
        final_image_cropped = rotated_image[:valid_height, :valid_width]

        # Create a mask where pixels aren't black
        mask = (final_image_cropped[:, :, 0] != 0) & (final_image_cropped[:, :, 1] != 0) & (final_image_cropped[:, :, 2] != 0)

        # Paste the image fragment on the painting using the mask
        painting[y_1:y_2, x_1:x_2][mask] = final_image_cropped[mask]
        """

    return painting


def show_image(image):
    # Show the image
    cv2.imshow('Image', image)
    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    # Then kill the image's window
    cv2.destroyAllWindows()


def get_painting(painting_path, black=False):
    painting = cv2.imread(painting_path)

    if black:
        height, width, _ = painting.shape
        # Create a black image of size y x with 3 channels (RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        return painting


def main():
    # Load the data about the images
    fragments_data = load_fragments(fragment_path)
    # Load the images according to the data collected
    fragments_images = load_images(fragments_data, fragment_directory)
    # Get the final painting or a blank image
    painting = get_painting(final_image_path, True)
    # Place the fragments on the painting
    final_image = place_fragments(painting, fragments_data, fragments_images)
    # Show the final painting with the fragments on it
    show_image(final_image)


if __name__ == '__main__':
   main()


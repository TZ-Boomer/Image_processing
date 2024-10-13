import cv2
import numpy as np
import math


# Paths and directories
FRAGMENT_DIRECTORY = "TBD.../TP1_resources/frag_eroded/"
FRAGMENT_PATH = "TBD.../TP1_resources/fragments.txt"
FINAL_IMAGE_PATH = "TBD.../TP1_resources/Michelangelo_ThecreationofAdam_1707x775.jpg"
SOLUTION_PATH = "TBD.../TP1_resources/solution.txt"

# Parameters for the solution file evaluation
DELTA_X = 1
DELTA_Y = 1
DELTA_ANGLE = 1


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
    print("Images loading...")
    images = []

    for i in range(len(fragments)):
        image_name = "frag_eroded_" + str(fragments[i][0]) + ".png"
        # Load an image from file
        image = cv2.imread(images_path + image_name)

        # Check if the image was loaded successfully
        if image is None:
            print("Error: Could not load image.")
        images.append(image)

    print("Images loaded done.")
    return images


def rotate_image(image, angle):
    height, width, _ = image.shape

    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def place_fragments(painting, fragments_data, fragments_images):
    for i, fragment_data in enumerate(fragments_data):
        x_i, y_i, angle = fragment_data[1], fragment_data[2], fragment_data[3]
        fragment = fragments_images[i]

        # Rotate the image
        rotated_fragment = rotate_image(fragment, angle)

        # Crop the black part around the fragment
        # Return a rectangle image still with black pixels along the curved parts of the fragment
        rotated_fragment_cropped = crop_black_contours(rotated_fragment)

        # Get the height and width of the rotated_fragment_cropped
        height, width, _ = rotated_fragment_cropped.shape

        # Compute top-left and bottom-right coordinates of the image in the painting
        x_1 = math.ceil(x_i - (width / 2))
        y_1 = math.ceil(y_i - (height / 2))
        x_2 = math.ceil(x_i + (width / 2))
        y_2 = math.ceil(y_i + (height / 2))

        # Create a mask for non-black pixels
        mask = (rotated_fragment_cropped > 3).any(axis=2)

        """ 
        There is still some black pixels remaining because the previous crop only reshaped the image as close as 
        possible of the fragment, still making a rectangle image. Inside the rectangle image, there is still some black 
        pixels we don't want to print on the painting. Thus, if a pixel is black, we don't print it thanks to the mask
        
        If we do not crop the image before computing its location, the fragment's image could be out of the painting.
        This is because of the large black border around the fragments.
        """
        # Paste the image fragment on the painting using the mask
        painting[y_1:y_2, x_1:x_2][mask] = rotated_fragment_cropped[mask]

        """
        Much slower version. Easy to implement.
        # "Paste" the image fragment on the painting
        for i in range(width):
            for j in range(height):
                # Add the new image pixels only if they aren't black, if they aren't the contour of the image
                if (rotated_fragment[j, i, 0] > 3) & (rotated_fragment[j, i, 1] > 3) & (rotated_fragment[j, i, 2] > 3):
                    painting[y_1 + j, x_1 + i] = rotated_fragment[j, i]
        """

    return painting


def crop_black_contours(image):
    # Convert to grayscale and create a binary mask where black pixels are 0 and non-black are 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours from the binary mask and get the bounding box as a rectangle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using the bounding box
    return image[y:y + h, x:x + w]


def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_painting(painting_path, black=False):
    painting = cv2.imread(painting_path)

    if black:
        height, width, _ = painting.shape
        # Create a black image of size y x with 3 channels (RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        return painting


def get_pixels_count(data, fragment_directory):
    fragment_pixels = {}

    for fragment in data:
        path = fragment_directory + f"frag_eroded_{fragment[0]}.png"
        image = cv2.imread(path, -1) # Load with alpha channel "-1"
        alpha_channel = image[:, :, 3]
        non_transparent_pixels = np.count_nonzero(alpha_channel > 0)
        fragment_pixels[fragment[0]] = non_transparent_pixels

    return fragment_pixels


def compute_solution_precision(fragments_data, solution_data, fragment_directory):
    well_placed = [] # Well-placed fragments
    miss_placed = [] # Miss-placed fragments
    wrong_fragments = [] # Fragments that shouldn't be placed
    solution_index = 0

    # Get the number of pixels for each fragment in the data files
    solution_pixels = get_pixels_count(solution_data, fragment_directory)
    frag_data_pixels = get_pixels_count(fragments_data, fragment_directory)

    # We assume that fragments in solution file are sorted in ascending order
    for fragment in fragments_data:
        # If we went through all solution fragments, break
        if solution_index >= len(solution_data):
            break

        # If the solution fragment ID is smaller (wrong) than the right one, go to the next in solution
        while fragment[0] > solution_data[solution_index][0]:
            wrong_fragments.append(solution_pixels.get(solution_data[solution_index][0]))
            solution_index += 1

        # If the solution fragment ID is bigger than the right one, go to the next in frag data
        if fragment[0] < solution_data[solution_index][0]:
            continue

        # If the two fragments have the same ID, check if it is well-placed (more or less delta)
        if fragment[0] == solution_data[solution_index][0]: # If the two ids are the same
            if ((abs(fragment[1] - solution_data[solution_index][1])) < DELTA_X and
                    (abs(fragment[2] - solution_data[solution_index][2])) < DELTA_Y and
                    (abs(fragment[3] - solution_data[solution_index][3])) < DELTA_ANGLE):
                well_placed.append(solution_pixels.get(solution_data[solution_index][0]))
            else:
                miss_placed.append(solution_pixels.get(solution_data[solution_index][0]))

        solution_index += 1

    total_fragments = len(well_placed) + len(miss_placed) + len(wrong_fragments)
    if total_fragments != len(solution_data):
        raise ValueError(f"Expected {len(solution_data)} fragments but got {total_fragments}. Check solution data.")

    # Compute the precision
    precision = (np.sum(well_placed) - np.sum(wrong_fragments)) / (sum(frag_data_pixels.values()))

    return precision


def image_reconstruction(fragment_path, fragment_directory, final_image_path):
    print("Image reconstruction...")
    # Load the data about the images
    fragments_data = load_fragments(fragment_path)
    # Load the images according to the data collected
    fragments_images = load_images(fragments_data, fragment_directory)
    # Get the final painting or a black image
    painting = get_painting(final_image_path, True)
    # Place the fragments on the painting
    final_image = place_fragments(painting, fragments_data, fragments_images)

    print("Painting with fragments placed")
    # Show the final painting with the fragments on it
    show_image(final_image)

    return fragments_data


def evaluate_solution(fragments_data, solution_path, fragment_directory):
    print("\nThe fragments in solution.txt must be sorted by their IDs in ascending order.\n")
    # Load the data in the solution file
    solution_data = load_fragments(solution_path)

    print(f"The precision of : {solution_path}")
    print(f"is : {compute_solution_precision(fragments_data, solution_data, fragment_directory) * 100:.4f}%")


def main():
    fragments_data = image_reconstruction(FRAGMENT_PATH, FRAGMENT_DIRECTORY, FINAL_IMAGE_PATH)
    evaluate_solution(fragments_data, SOLUTION_PATH, FRAGMENT_DIRECTORY)


if __name__ == '__main__':
   main()

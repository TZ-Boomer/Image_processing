import cv2
import numpy as np
from math import *


IMAGE_PATH = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP2_resources/images/four.png"


""" Hough parameters """
DELTA_R = 1
DELTA_C = 1
DELTA_RAD = 1
MIN_R = 0
MIN_C = 0
MIN_RAD = 5
N_CIRCLES = 2
THRESHOLD = 0.2
LOCAL_MAX_KERNEL_SIZE = 3
IMAGE_REDUCTION = 2 # Image divided by 2 at each iteration


def main():
    # Load the image
    image = cv2.imread(IMAGE_PATH) #cv2.IMREAD_GRAYSCALE

    # Get the edges image
    edges_image = get_edges_image(image, threshold_ratio=THRESHOLD)

    display_image(edges_image)

    image_reduction(edges_image, image)

    #hough_method(edges_image, image)


def resize_image(image, scale_factor=0.5):
    # Calculate new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Resize the image
    new_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return new_image


def image_reduction(edges_image, original_image):
    edges_images = []
    original_images = []
    local_maxima_coords = []

    edges_images.append(edges_image)
    original_images.append(original_image)

    for i in range(IMAGE_REDUCTION):
        # We take the previous resize [-1] and resize it again
        edges_images.append(resize_image(edges_images[-1], 0.5))
        original_images.append(resize_image(original_images[-1], 0.5))

    print("Number of images : ", len(edges_images))

    for i in range(IMAGE_REDUCTION, -1, -1):
        print("============================== Image reduction iteration : ", i)
        print("Number of circles coordinates : ", len(local_maxima_coords))
        # Multiply the coordinates by 2
        local_maxima_coords = [[row[0] * 2, row[1] * 2, row[2] * 4, row[3]] for row in local_maxima_coords]

        if i == IMAGE_REDUCTION:
            # First iteration, we go through the whole image
            local_max = hough_method(edges_images[i])
            print("LEN local max : ", len(local_max))
            local_maxima_coords.extend(local_max)
        else:
            # We only compute the circles in range [MIN_RAD ; 2 * MIN_RAD], because 2 * MIN_RAD was our previous MIN_RAD
            local_max = hough_method(edges_images[i], 2 * MIN_RAD)
            print("LEN local max : ", len(local_max))
            local_maxima_coords.extend(local_max)

        # Create a copy of the image, draw the circles and display it
        image_copy = original_images[i].copy()
        draw_circles(image_copy, local_maxima_coords)
        display_image(image_copy)

        print("local_maxima_coords : ", local_maxima_coords)





def display_image(image):
    # Display the edges
    cv2.imshow('Image', image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_edges_image(image, gaussian_ksize=(5, 5), threshold_ratio=0.1):
    # Step 1: Apply Gaussian blur to reduce noise (optional but recommended)
    blurred_image = cv2.GaussianBlur(image, gaussian_ksize, 0)

    # Step 2: Apply Sobel filters to compute the gradients in the x and y directions
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Step 3: Compute the gradient magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Step 4: Normalize the magnitude to the range [0, 255] for better visualization
    magnitude_normalized = np.uint8(np.clip(magnitude / np.max(magnitude) * 255, 0, 255))

    # Step 5: Threshold to create the final edge image (binary)
    _, edges_image = cv2.threshold(magnitude_normalized, threshold_ratio * 255, 255, cv2.THRESH_BINARY)

    return edges_image


def get_edges_coordinates(edges_image):
    # Ensure the input is a 2D binary image (single-channel)
    if len(edges_image.shape) != 2 or edges_image.shape[2] != 1:
        edges_image = cv2.cvtColor(edges_image, cv2.COLOR_BGR2GRAY)

    # Find the coordinates of edge pixels (non-zero pixels in the binary image)
    edge_coordinates = cv2.findNonZero(edges_image)

    # Convert the coordinates to a list of tuples (x, y)
    edge_coordinates = [(point[0][0], point[0][1]) for point in edge_coordinates]

    return edge_coordinates


def populate_accumulator(edges_image, max_radius):
    # Get the dimensions of the image
    height, width = edges_image.shape[:2]

    r_max = height - 1
    c_max = width - 1

    n_c = floor(((c_max - MIN_C) / DELTA_C) + 1)  # The number of column discretize
    n_r = floor(((r_max - MIN_R) / DELTA_R) + 1)  # The number of rows discretize

    if max_radius == -1: # We compute all the possible circle for the whole image between MIN_RAD and diagonal
        diagonal = np.sqrt(width ** 2 + height ** 2)
        n_rad = floor(((diagonal - MIN_RAD) / DELTA_R) + 1)
    else: # We compute the possible circles between MIN_RAD and max_radius
        diagonal = max_radius
        n_rad = floor(((diagonal - MIN_RAD) / DELTA_R) + 1)

    print("n_r_values : ", n_r)
    print("n_c_values : ", n_c)
    print("n_rad_values : ", n_rad)
    print("diagonal : ", diagonal)
    accumulator = np.zeros((n_c, n_r, n_rad))

    edge_coordinates = get_edges_coordinates(edges_image)

    for edge in edge_coordinates:
        edge_x, edge_y = edge

        for c_idx in range(width):
            for r_idx in range(height):
                # If the point in the image is too close to the edge, we do not compute the associated circle
                circle_radius = compute_pixels_distance(c_idx, r_idx, edge_x, edge_y)
                if MIN_RAD <= circle_radius < diagonal:
                    acc_c_idx = int((c_idx - MIN_C) / DELTA_C)
                    acc_r_idx = int((r_idx - MIN_R) / DELTA_R)
                    acc_rad_idx = int((circle_radius - MIN_RAD) / DELTA_RAD)

                    if 0 <= acc_c_idx < n_c and 0 <= acc_r_idx < n_r and 0 <= acc_rad_idx < n_rad:
                        accumulator[acc_c_idx][acc_r_idx][acc_rad_idx] += (1 / (2 * np.pi * circle_radius))
                    else:
                        raise Exception(f"Unexpected index value for the accumulator "
                                        f"acc_c_idx : {acc_c_idx}, acc_r_idx : {acc_r_idx}, acc_rad_idx : {acc_rad_idx}")

    return accumulator


def get_local_maximum(accumulator):
    n_c, n_r, n_rad = accumulator.shape

    padding = int(LOCAL_MAX_KERNEL_SIZE / 2)
    kernel = list(range(-padding, padding + 1))

    # Initialize an empty list to store the coordinates of local maxima
    local_maxima_coords = []

    # Loop through each element in the 3D array, avoiding the borders
    for x in range(padding, n_c - padding):
        for y in range(padding, n_r - padding):
            for z in range(padding, n_rad - padding):
                # Get the current value
                current_value = accumulator[x, y, z]

                # Flag to check if this is a local maximum
                is_local_maximum = True

                # Loop through the neighboring cells
                for dx in kernel:
                    for dy in kernel:
                        for dz in kernel:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue  # Skip the current cell itself

                            # Compare with neighboring cell
                            neighbor_value = accumulator[x + dx, y + dy, z + dz]
                            if neighbor_value >= current_value:
                                is_local_maximum = False
                                break  # Stop checking neighbors if one is greater or equal
                        if not is_local_maximum:
                            break
                    if not is_local_maximum:
                        break

                # If it's a local maximum, save its coordinates
                if is_local_maximum:
                    acc_value = accumulator[x, y, z]
                    local_maxima_coords.append((x, y, z, acc_value))

    # Sort the local_maxima_coords in descending order based on the accumulator value
    sorted_local_maxima_coords = sorted(local_maxima_coords, key=lambda x: x[3], reverse=True)
    sorted_local_maxima_coords = sorted_local_maxima_coords[:N_CIRCLES]

    return sorted_local_maxima_coords


def draw_circles(original_image, sorted_local_maxima_coords):
    color = (0, 255, 0)  # Color in BGR (Green in this case)
    thickness = 1  # Line thickness (-1 to fill the circle)

    n_circles = len(sorted_local_maxima_coords)

    # Drawing circles
    for i in range(n_circles):
        x_idx, y_idx, radius_idx = sorted_local_maxima_coords[i][:3]
        x = int((x_idx * DELTA_C) + MIN_C)
        y = int((y_idx * DELTA_R) + MIN_R)
        radius = int((radius_idx * DELTA_RAD) + MIN_RAD)

        cv2.circle(original_image, (x, y), radius, color, thickness)


def hough_method(edges_image, max_radius=-1):
    accumulator = populate_accumulator(edges_image, max_radius)

    sorted_local_maxima_coords = get_local_maximum(accumulator)

    return sorted_local_maxima_coords



def compute_pixels_distance(p1x, p1y, p2x, p2y):
    return sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)


if __name__ == '__main__':
   main()

import cv2
import numpy as np
from math import *
from TPs.TP2 import paths


IMAGE_PATH = paths.IMAGE_PATH

""" Hough parameters """
DELTA_R = 1
DELTA_C = 1
DELTA_RAD = 1
MIN_R = 0
MIN_C = 0
MIN_RAD = 5
N_CIRCLES = 1


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



def hough_method(edges_image, original_image):
    # Get the dimensions of the image
    height, width = edges_image.shape[:2]

    r_max = height - 1
    c_max = width - 1

    n_c = floor(((c_max - MIN_C) / DELTA_C) + 1) # The number of column discretize
    n_r = floor(((r_max - MIN_R) / DELTA_R) + 1) # The number of rows discretize
    diagonal = np.sqrt(width ** 2 + height ** 2)
    n_rad = floor(((diagonal - MIN_RAD) / DELTA_R) + 1)
    print("n_r_values : ", n_r)
    print("n_c_values : ", n_c)
    print("n_rad_values : ", n_rad)
    print("diagonal : ", diagonal)

    accumulator = np.zeros((n_c, n_r, n_rad))

    # Assuming you have already obtained the edges_image from the Sobel and thresholding steps
    edge_coordinates = get_edges_coordinates(edges_image)

    # Vote in the accumulator
    for edge in edge_coordinates:
        edge_x, edge_y = edge
        for r in range(MIN_RAD, n_r):
            for c in range(MIN_RAD, n_c):
                radius = compute_pixels_distance(c, r, edge_x, edge_y)
                if MIN_RAD <= radius < diagonal:
                    acc_r_idx = int((r - MIN_R) / DELTA_R)
                    acc_c_idx = int((c - MIN_C) / DELTA_C)
                    acc_rad_idx = int((radius - MIN_RAD) / DELTA_RAD)
                    if 0 <= acc_c_idx < n_c and 0 <= acc_r_idx < n_r and 0 <= acc_rad_idx < n_rad:
                        accumulator[acc_c_idx][acc_r_idx][acc_rad_idx] += 1

    # Local maxima detection
    local_maxima_coords = []
    for x in range(1, n_r - 1):
        for y in range(1, n_c - 1):
            for z in range(1, n_rad - 1):
                current_value = accumulator[x, y, z]
                is_local_maximum = True
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            if accumulator[x + dx, y + dy, z + dz] >= current_value:
                                is_local_maximum = False
                                break
                        if not is_local_maximum:
                            break
                    if not is_local_maximum:
                        break
                if is_local_maximum:
                    local_maxima_coords.append((x, y, z, current_value))

    # Sort and draw the top N circles
    sorted_local_maxima_coords = sorted(local_maxima_coords, key=lambda x: x[3], reverse=True)
    color = (0, 255, 0)  # Green color for circles
    thickness = 1
    for i in range(N_CIRCLES):
        x_idx, y_idx, radius_idx = sorted_local_maxima_coords[i][:3]
        x = (x_idx * DELTA_C) + MIN_C
        y = (y_idx * DELTA_R) + MIN_R
        radius = (radius_idx * DELTA_RAD) + MIN_RAD
        center_coordinates = (int(x), int(y))
        cv2.circle(original_image, center_coordinates, int(radius), color, thickness)

    # Display the final image
    display_image(original_image)


def compute_pixels_distance(p1x, p1y, p2x, p2y):
    return sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)


def main():
    # Load the image
    image = cv2.imread(IMAGE_PATH) #cv2.IMREAD_GRAYSCALE

    # Get the edges image
    edges_image = get_edges_image(image, threshold_ratio=0.1)

    #display_image(edges_image)

    hough_method(edges_image, image)



if __name__ == '__main__':
   main()

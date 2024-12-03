import numpy as np
from TPs.TP2 import paths
import matplotlib.pyplot as plt
import cv2
import sys
from math import floor, sqrt
from scipy.ndimage import maximum_filter


IMAGE_PATH = paths.IMAGE_PATH


""" Hough parameters """
DELTA_R = 1
DELTA_C = 1
DELTA_RAD = 1
MIN_R = 0
MIN_C = 0
MIN_RAD = 5

N_CIRCLES = 8 # Per iteration, after the first iteration there is a low tolerance to add new circles (MIN_DETECT_LEVEL)
FILTER_EDGES_THRESHOLD = 0.3
# Size of the kernel (cube) that avoid multiple similar circles (same center and radius), depend on the scale of the image
LOCAL_MAX_KERNEL_SIZE = 35

IMAGE_REDUCTION_LEVELS = 3 # Image divided by SCALE_FACTOR at each iteration
SCALE_FACTOR = 2
MIN_DETECT_LEVEL = 0.5
CIRCLE_CENTER_UPDATE_RANGE = 1
CIRCLE_RADIUS_UPDATE_RANGE = 11


def main():
    # Load the image in grayscale
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Error: Unable to load image at {IMAGE_PATH}')
        sys.exit(1)

    # Perform image reduction and circle detection
    image_reduction(image)


def image_reduction(original_image):
    # Convert the original image to BGR color space for drawing colored circles
    final_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR)
    original_images = [cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)]

    # Extract the edges image
    sobel_data = sobel_filter(original_image, threshold_ratio=FILTER_EDGES_THRESHOLD)
    edges_images = [sobel_data['edges_image']]

    local_maxima_accumulator = []

    # Create reduced images
    for i in range(IMAGE_REDUCTION_LEVELS):
        edges_images.append(resize_image(edges_images[-1]))
        original_images.append(resize_image(original_images[-1]))

    print("Number of image scales :", len(edges_images))

    # Iterate through the scales
    for i in range(IMAGE_REDUCTION_LEVELS, -1, -1):
        print(f"===== Iteration number : {i}")

        # Scale up the coordinates and radius
        if local_maxima_accumulator:
            local_maxima_accumulator = scale_coordinates(local_maxima_accumulator)

        # Apply Sobel filter to get gradient direction and other data
        sobel_data = sobel_filter(original_images[i], threshold_ratio=FILTER_EDGES_THRESHOLD)
        gradient_direction = sobel_data['gradient_direction']

        display_sobel_filter_results(original_images[i], edges_images[i], sobel_data)

        if i == IMAGE_REDUCTION_LEVELS:  # Initial case on the whole image
            initial_circles = hough_method(
                edges_images[i],
                image_gradient_direction=gradient_direction,
                current_iteration = i,
                max_radius=-1,
                local_maxima_accumulator=None
            )
            local_maxima_accumulator.extend(initial_circles)
        else:
            # Detect small circles
            new_circles = hough_method(
                edges_images[i],
                image_gradient_direction=gradient_direction,
                current_iteration=i,
                max_radius=MIN_RAD * 2
            )

            # Update old circles coordinates
            updated_circles = hough_method(
                edges_images[i],
                image_gradient_direction=gradient_direction,
                current_iteration=i,
                max_radius=-1,
                local_maxima_accumulator=local_maxima_accumulator
            )
            local_maxima_accumulator = updated_circles

            # Keep circles with high probability
            best_new_circles = [circle for circle in new_circles if circle[3] > MIN_DETECT_LEVEL]
            local_maxima_accumulator.extend(best_new_circles)

        print("Coordinates :", local_maxima_accumulator)
        draw_circles(original_images[i], local_maxima_accumulator)
        display_images(
            [cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB)],
            titles=[f'Detected Circles at Scale {i}'],
            figsize=(8, 8)
        )

    draw_circles(final_image, local_maxima_accumulator)
    display_images(
        [cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)],
        titles=['Final Detected Circles'],
        figsize=(10, 10)
    )


def hough_method(edges_image, image_gradient_direction, current_iteration, max_radius=-1, local_maxima_accumulator=None):
    if local_maxima_accumulator is None:
        accumulator = populate_accumulator(edges_image, image_gradient_direction, max_radius)
        sorted_local_maxima_accumulator = get_local_maxima(accumulator, current_iteration)
    else:
        sorted_local_maxima_accumulator = update_accumulator(edges_image, local_maxima_accumulator)

    print("HOUGH LEN : ", len(sorted_local_maxima_accumulator))

    return sorted_local_maxima_accumulator


def populate_accumulator(edges_image, image_gradient_direction, max_radius):
    print("_____ Populating accumulator...")
    image_height, image_width = edges_image.shape[:2]

    r_max = image_height - 1
    c_max = image_width - 1

    n_col = floor(((c_max - MIN_C) / DELTA_C) + 1)
    n_row = floor(((r_max - MIN_R) / DELTA_R) + 1)
    diagonal = np.sqrt(image_width ** 2 + image_height ** 2) if max_radius == -1 else max_radius
    n_rad = floor(((diagonal - MIN_RAD) / DELTA_RAD) + 1)

    print(f"Accumulator size : {n_col} x {n_row} x {n_rad}")
    accumulator = np.zeros((n_col, n_row, n_rad))

    edge_coordinates = get_edges_coordinates(edges_image)

    for edge_x, edge_y in edge_coordinates:
        theta = image_gradient_direction[edge_y, edge_x] + np.pi  # Reverse direction

        for radius in range(MIN_RAD, int(diagonal), 1):
            center_x = round(edge_x + radius * np.cos(theta))
            center_y = round(edge_y + radius * np.sin(theta))

            if 0 <= center_x < image_width and 0 <= center_y < image_height:
                acc_c_idx = int((center_x - MIN_C) / DELTA_C)
                acc_r_idx = int((center_y - MIN_R) / DELTA_R)
                acc_rad_idx = int((radius - MIN_RAD) / DELTA_RAD)
                accumulator[acc_c_idx, acc_r_idx, acc_rad_idx] += (1 / (2 * np.pi * radius))

    return accumulator


def update_accumulator(edges_image, local_maxima_accumulator):
    print("_____________ Updating accumulator...")
    # Get the dimensions of the image
    height, width = edges_image.shape[:2]

    new_local_maxima_accumulator = []

    r_max = height - 1
    c_max = width - 1

    n_col = floor(((c_max - MIN_C) / DELTA_C) + 1)  # The number of column discretize
    n_row = floor(((r_max - MIN_R) / DELTA_R) + 1)  # The number of rows discretize
    diagonal = np.sqrt(width ** 2 + height ** 2)
    n_rad = floor(((diagonal - MIN_RAD) / DELTA_R) + 1)

    print("===== Accumulator size : ")
    print("n_row : ", n_row)
    print("n_col : ", n_col)
    print("n_rad : ", n_rad)
    print("diagonal : ", diagonal)
    print("number of elements : ", n_row * n_col * n_rad)
    accumulator = np.zeros((n_col, n_row, n_rad))

    edge_coordinates = get_edges_coordinates(edges_image)
    for circle in local_maxima_accumulator:
        cir_x, cir_y, cir_rad, _ = circle

        # +1 because the upper limit is excluded from the interval
        c_idx_range = range(
            max(int(cir_x) - CIRCLE_CENTER_UPDATE_RANGE, MIN_C),
            min(int(cir_x) + CIRCLE_CENTER_UPDATE_RANGE + 1, n_col + 1)
        )
        r_idx_range = range(
            max(int(cir_y) - CIRCLE_CENTER_UPDATE_RANGE, MIN_R),
            min(int(cir_y) + CIRCLE_CENTER_UPDATE_RANGE + 1, n_row + 1)
        )
        rad_idx_range = range(
            max(int(cir_rad) - CIRCLE_RADIUS_UPDATE_RANGE, MIN_RAD),
            min(int(cir_rad) + CIRCLE_RADIUS_UPDATE_RANGE + 1, n_rad + 1)
        )

        for col_idx in c_idx_range:
            for row_idx in r_idx_range:
                for rad_idx in rad_idx_range:
                    # Precompute distances to edge points within the max radius range
                    nearby_edges = [
                        (edge_x, edge_y) for edge_x, edge_y in edge_coordinates
                        if abs(edge_x - col_idx) <= rad_idx_range[-1] and abs(edge_y - row_idx) <= rad_idx_range[-1]
                    ]
                    # For each edge check if it is on the perimeter of the circle
                    for edge_x, edge_y in nearby_edges:

                        circle_radius = compute_pixels_distance(col_idx, row_idx, edge_x, edge_y)
                        # Update the accumulator if the edged is on the perimeter of the circle
                        # The values aren't exact so add a boundary
                        if rad_idx - 0.5 <= circle_radius <= rad_idx + 0.5:
                            acc_c_idx = round((col_idx - MIN_C) / DELTA_C)
                            acc_r_idx = round((row_idx - MIN_R) / DELTA_R)
                            acc_rad_idx = round((rad_idx - MIN_RAD) / DELTA_RAD)

                            if 0 <= acc_c_idx < n_col and 0 <= acc_r_idx < n_row and 0 <= acc_rad_idx < n_rad:
                                accumulator[acc_c_idx][acc_r_idx][acc_rad_idx] += (1 / (2 * np.pi * rad_idx))
                            else:
                                raise Exception(f"Unexpected index value for the accumulator "
                                                f"acc_c_idx : {acc_c_idx}, acc_r_idx : {acc_r_idx}, acc_rad_idx : {acc_rad_idx}")

        max_value = np.max(accumulator)
        max_indices = np.unravel_index(np.argmax(accumulator), accumulator.shape)

        # Combine coordinates and values
        row = (max_indices[0], max_indices[1], max_indices[2], max_value)
        # Add the updated circle to the final list
        new_local_maxima_accumulator.append(row)
        # Reinitialize accumulator for the next circle
        accumulator = np.zeros((n_col, n_row, n_rad))

    print("LEN new_local_maxima_accumulator : ", len(new_local_maxima_accumulator))

    return new_local_maxima_accumulator


def display_sobel_filter_results(original_image, edges_image, sobel_data):
    # Display images at this scale
    images_to_show = [
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
        edges_image,
        sobel_data['sobel_x_normalized'],
        sobel_data['sobel_y_normalized'],
        sobel_data['gradient_magnitude_normalized'],
        sobel_data['gradient_direction']
    ]
    titles = [
        'Original Image',
        'Edges Image',
        'Sobel X',
        'Sobel Y',
        'Gradient Magnitude',
        'Gradient Direction'
    ]

    display_images(images_to_show, titles=titles, figsize=(15, 5))

def scale_coordinates(local_maxima_accumulator):
    # Convert accumulator indices to circle parameters
    local_maxima_accumulator = [
        [
            (coord[0] * DELTA_C) + MIN_C,  # x-coordinate
            (coord[1] * DELTA_R) + MIN_R,  # y-coordinate
            (coord[2] * DELTA_RAD) + MIN_RAD,  # radius
            coord[3]  # Likeliness of the circle
        ]
        for coord in local_maxima_accumulator
    ]

    # Scale up the circle parameters
    local_maxima_accumulator = [
        [
            coord[0] * SCALE_FACTOR,  # Scaled x-coordinate
            coord[1] * SCALE_FACTOR,  # Scaled y-coordinate
            coord[2] * SCALE_FACTOR,  # Scaled radius
            coord[3]
        ]
        for coord in local_maxima_accumulator
    ]

    # Map back to accumulator indices for the new scale
    local_maxima_accumulator = [
        [
            int((coord[0] - MIN_C) / DELTA_C),  # x_idx
            int((coord[1] - MIN_R) / DELTA_R),  # y_idx
            int((coord[2] - MIN_RAD) / DELTA_RAD),  # radius_idx
            coord[3]
        ]
        for coord in local_maxima_accumulator
    ]

    return local_maxima_accumulator


def get_local_maxima(accumulator, current_iteration):
    # Apply maximum filter
    if current_iteration > 0:
        neighborhood_size = round(LOCAL_MAX_KERNEL_SIZE / (SCALE_FACTOR * current_iteration))
    else:
        neighborhood_size = LOCAL_MAX_KERNEL_SIZE
    print("neighborhood_size : ", neighborhood_size)
    data_max = maximum_filter(accumulator, size=neighborhood_size, mode='constant')
    maxima = (accumulator == data_max)
    diff = (data_max > 0)
    maxima[diff == 0] = 0

    # Get coordinates of local maxima
    local_max_coords = np.argwhere(maxima)
    local_max_values = accumulator[maxima]

    # Combine coordinates and values
    local_maxima_accumulator = [
        (coord[0], coord[1], coord[2], val)
        for coord, val in zip(local_max_coords, local_max_values)
    ]

    # Sort and select top N circles
    sorted_local_maxima_accumulator = sorted(local_maxima_accumulator, key=lambda x: x[3], reverse=True)

    return sorted_local_maxima_accumulator[:N_CIRCLES]

# Way to slow
def get_local_maxima_v0(accumulator):
    n_col, n_row, n_rad = accumulator.shape

    padding = int(LOCAL_MAX_KERNEL_SIZE / 2)
    kernel = list(range(-padding, padding + 1))

    # Initialize an empty list to store the coordinates of local maxima
    local_maxima_accumulator = []

    # Loop through each element in the 3D array, avoiding the borders
    for x in range(padding, n_col - padding):
        for y in range(padding, n_row - padding):
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
                    local_maxima_accumulator.append((x, y, z, acc_value))

    # Sort the local_maxima_accumulator in descending order based on the accumulator value
    sorted_local_maxima_accumulator = sorted(local_maxima_accumulator, key=lambda x: x[3], reverse=True)
    sorted_local_maxima_accumulator = sorted_local_maxima_accumulator[:N_CIRCLES]

    return sorted_local_maxima_accumulator


def resize_image(image):
    new_width = int(image.shape[1] * (1 / SCALE_FACTOR))
    new_height = int(image.shape[0] * (1 / SCALE_FACTOR))
    new_dimensions = (new_width, new_height)
    new_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return new_image


def display_images(images, titles=None, figsize=(10, 10)):
    n_images = len(images)
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, n_images, i + 1)
        if len(image.shape) == 2:
            if i == len(images) - 1:  # Gradient direction uses a colormap for angles
                plt.imshow(images[i], cmap='hsv')  # Hue for direction
            else:
                plt.imshow(images[i], cmap='gray')  # Grayscale for other visualizations
        else:
            plt.imshow(image)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_edges_coordinates(edges_image):
    # Ensure the input is a 2D binary image (single-channel)
    if len(edges_image.shape) != 2:
        edges_image = cv2.cvtColor(edges_image, cv2.COLOR_BGR2GRAY)

    # Find the coordinates of edge pixels (non-zero pixels in the binary image)
    edge_coordinates = cv2.findNonZero(edges_image)

    if edge_coordinates is not None:
        # Convert the coordinates to a list of tuples (x, y)
        edge_coordinates = [(point[0][0], point[0][1]) for point in edge_coordinates]
    else:
        edge_coordinates = []

    return edge_coordinates


def draw_circles(image, circles):
    # If image is grayscale, convert it to BGR color space
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    color = (0, 255, 0)  # Green color in BGR
    thickness = 2

    for x_idx, y_idx, radius_idx, _ in circles:
        x = int((x_idx * DELTA_C) + MIN_C)
        y = int((y_idx * DELTA_R) + MIN_R)
        radius = int((radius_idx * DELTA_RAD) + MIN_RAD)
        cv2.circle(image, (x, y), radius, color, thickness)


def sobel_filter(image, threshold_ratio, gaussian_ksize=(5, 5)):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply Gaussian blur to reduce noise (optional but recommended)
    blurred_image = cv2.GaussianBlur(gray_image, gaussian_ksize, 0)

    # Apply the Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize Sobel outputs for visualization (scale to 0-255)
    sobel_x_normalized = cv2.convertScaleAbs(sobel_x)
    sobel_y_normalized = cv2.convertScaleAbs(sobel_y)
    # Normalize the magnitude to the range [0, 255] for better visualization
    gradient_magnitude_normalized = np.uint8(np.clip(gradient_magnitude / np.max(gradient_magnitude) * 255, 0, 255))

    # Threshold to create the final edge image (binary)
    _, edges_image = cv2.threshold(gradient_magnitude_normalized, threshold_ratio * 255, 255, cv2.THRESH_BINARY)

    # Return all data needed
    return {
        'edges_image': edges_image,
        'gradient_direction': gradient_direction,
        'sobel_x_normalized': sobel_x_normalized,
        'sobel_y_normalized': sobel_y_normalized,
        'gradient_magnitude_normalized': gradient_magnitude_normalized,
        'gray_image': gray_image
    }


def compute_pixels_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


if __name__ == '__main__':
    main()

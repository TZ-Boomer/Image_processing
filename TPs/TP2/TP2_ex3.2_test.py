import cv2
import numpy as np
from math import floor, sqrt, cos, sin, pi
from scipy.ndimage import maximum_filter


IMAGE_PATH = "C:/Users/julie/Desktop/All/Important/Polytech/Inge_3/Traitement_d_image/TP2_resources/images/four.png"


""" Hough parameters """
DELTA_RAD = 1
MIN_RAD = 5
MAX_RAD = 100
N_CIRCLES = 2
FILTER_EDGES_THRESHOLD = 0.2
LOCAL_MAX_KERNEL_SIZE = 3

IMAGE_REDUCTION = 2  # Nombre de fois que l'image est réduite de moitié

def main():
    # Load the image in grayscale
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    # Get the edges image
    edges_image = get_edges_image(image, threshold_ratio=FILTER_EDGES_THRESHOLD)

    display_image(edges_image, 'Edges Image')

    image_reduction(edges_image, image)

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
        # We take the previous resized image and resize it again
        edges_images.append(resize_image(edges_images[-1], 0.5))
        original_images.append(resize_image(original_images[-1], 0.5))

    print("Number of images : ", len(edges_images))

    for i in range(IMAGE_REDUCTION, -1, -1):
        print("============================== Image reduction iteration : ", i)
        print("Number of circles coordinates : ", len(local_maxima_coords))
        # Multiply the coordinates by 2 to upscale to the higher resolution
        if local_maxima_coords:
            local_maxima_coords = [(x * 2, y * 2, radius * 2) for x, y, radius in local_maxima_coords]

        # Determine the minimum and maximum radii for this level
        scale = 2 ** i
        min_rad = max(int(MIN_RAD / scale), 1)
        max_rad = int(MAX_RAD / scale)

        print(f"Min radius: {min_rad}, Max radius: {max_rad}")

        # Detect circles at the current level
        local_max = hough_method(edges_images[i], min_rad, max_rad)
        print("LEN local max : ", len(local_max))
        local_maxima_coords.extend(local_max)

        # Create a copy of the image, draw the circles and display it
        image_copy = original_images[i].copy()
        image_color = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        draw_circles(image_color, local_maxima_coords)
        display_image(image_color, f'Detected Circles at Level {i}')

        print("local_maxima_coords : ", local_maxima_coords)

def display_image(image, window_name='Image'):
    # Display the image
    cv2.imshow(window_name, image)
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

def hough_method(edges_image, min_rad, max_rad):
    accumulator = populate_accumulator(edges_image, min_rad, max_rad)

    sorted_local_maxima_coords = get_local_maximum(accumulator, N_CIRCLES, min_rad)

    return sorted_local_maxima_coords

def populate_accumulator(edges_image, min_radius, max_radius):
    # Get the dimensions of the image
    height, width = edges_image.shape[:2]

    # Define the range of radii
    radii = np.arange(min_radius, max_radius + 1, DELTA_RAD)

    # Initialize the accumulator array
    accumulator = np.zeros((height, width, len(radii)), dtype=np.uint64)

    # Get indices of edge pixels
    edge_points = np.argwhere(edges_image > 0)

    # Precompute the angles
    angles = np.deg2rad(np.arange(0, 360, 5))  # Adjust angle step as needed
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Vote in the accumulator
    for r_idx, radius in enumerate(radii):
        # Compute circle perimeter offsets for this radius
        circle_perimeter_x = radius * cos_angles
        circle_perimeter_y = radius * sin_angles

        for x0, y0 in edge_points:
            # Potential centers for this edge point and radius
            a = x0 - circle_perimeter_y
            b = y0 - circle_perimeter_x

            # Round to nearest integer pixel values
            a = np.round(a).astype(np.int32)
            b = np.round(b).astype(np.int32)

            # Keep centers within the image bounds
            valid_idx = (a >= 0) & (a < height) & (b >= 0) & (b < width)
            a = a[valid_idx]
            b = b[valid_idx]

            # Increment the accumulator
            accumulator[a, b, r_idx] += 1

    return accumulator

def get_local_maximum(accumulator, n_circles, min_radius):
    # Apply a maximum filter to find local maxima
    filtered_acc = maximum_filter(accumulator, size=LOCAL_MAX_KERNEL_SIZE)
    local_max = (accumulator == filtered_acc) & (accumulator > 0)

    # Get the coordinates of the local maxima
    coords = np.argwhere(local_max)
    values = accumulator[local_max]

    # Sort the local maxima by their accumulator values
    sorted_indices = np.argsort(values)[::-1]
    top_indices = coords[sorted_indices[:n_circles]]

    # Convert indices to coordinates
    circles = []
    for idx in top_indices:
        a, b, r_idx = idx
        radius = min_radius + r_idx * DELTA_RAD
        circles.append((b, a, radius))  # Note: (x, y, radius)

    return circles

def draw_circles(image, circles):
    color = (0, 255, 0)  # Color in BGR (Green in this case)
    thickness = 2  # Line thickness

    # Drawing circles
    for x, y, radius in circles:
        cv2.circle(image, (int(x), int(y)), int(radius), color, thickness)

if __name__ == '__main__':
    main()

import math
import cv2
import os
import numpy as np
import re
import csv


def decompose_affine_no_scale(M):
    angle = math.atan2(M[1, 0], M[0, 0])
    posx = M[0, 2]
    posy = M[1, 2]
    angle_deg = math.degrees(angle)

    return posx, posy, angle_deg


def load_fragments(fragment_path):
    fragments = []
    with open(fragment_path, 'r') as file:
        for line in file:
            values = line.split()
            values = [int(values[0]), int(values[1]), int(values[2]), float(values[3])]
            fragments.append(values)
    return fragments


def load_images(image_directory):
    print("Images loading...")
    images = []

    for file_name in os.listdir(image_directory):
        file_path = os.path.join(image_directory, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Extract index from the filename using regex
                match = re.search(r'(\d+)', file_name)  # Looks for digits in the name
                index = int(match.group(1)) if match else None

                # Load the image
                img = cv2.imread(file_path)
                if img is not None:
                    images.append((index, img))
                else:
                    print(f"Error loading image {file_name}: File may be corrupted.")
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")

    print("Images loaded done.")

    return images


def get_painting(painting_path, black=False):
    painting = cv2.imread(painting_path)

    if painting is None:
        raise FileNotFoundError(f"Could not load the fresco {painting_path}.")

    if black:
        height, width, _ = painting.shape
        # Create a black image of size y x with 3 channels (RGB)
        return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        return painting


def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_and_compute(image, detector_n_features, detector_type="SIFT"):
    if image is None:
        return None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detector_type == "SIFT":
        detector = cv2.SIFT_create(nfeatures=detector_n_features)
    elif detector_type == "ORB":
        detector = cv2.ORB_create(nfeatures=detector_n_features)
    elif detector_type == "FAST":
        detector = cv2.FastFeatureDetector_create()
        return detector.detect(image, None), None
    else:
        raise ValueError("Unsupported detector type")

    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return keypoints, descriptors


def match_keypoints(desc1, desc2, method="FLANN", ratio_thresh=0.6):
    if desc1 is None or desc2 is None:
        return []

    if method == "FLANN":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    elif method == "BF":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError("Unsupported matching method")

    try:
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error as e:
        print(f"OpenCV error in knnMatch : {e}")
        return []

    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]

    return good_matches


def overlay_fragment_on_painting_no_scale(painting, fragment, M_affine):
    if fragment is None or painting is None:
        return painting

    h, w, _ = painting.shape

    # Warp using 2x3 affine matrix
    warped_fragment = cv2.warpAffine(fragment, M_affine, (w, h))

    # Assume black background in fragment => threshold
    fragment_gray = cv2.cvtColor(warped_fragment, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fragment_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Combine
    painting_bg = cv2.bitwise_and(painting, painting, mask=mask_inv)
    painting_fg = cv2.bitwise_and(warped_fragment, warped_fragment, mask=mask)
    final = cv2.add(painting_bg, painting_fg)

    return final


def get_pixels_count(data, fragment_directory):
    fragment_pixels = {}

    for fragment in data:
        path = fragment_directory + f"frag_eroded_{fragment[0]}.png"
        image = cv2.imread(path, -1) # Load with alpha channel "-1"
        alpha_channel = image[:, :, 3]
        non_transparent_pixels = np.count_nonzero(alpha_channel > 0)
        fragment_pixels[fragment[0]] = non_transparent_pixels

    return fragment_pixels


def compute_solution_precision(fragments_data, ground_truth_solution_data, fragment_directory, DELTA_X, DELTA_Y, DELTA_ANGLE):
    frag_well_placed = [] # Well-placed fragments
    wrong_frag = [] # Fragments that shouldn't be placed
    gd_solution_idx = 0
    frag_in = 0

    # Get the number of pixels for each fragment in the data files
    ground_truth_solution_pixels = get_pixels_count(ground_truth_solution_data, fragment_directory)
    fragments_data_pixels = get_pixels_count(fragments_data, fragment_directory)

    print("LEN FRAG DATA : ", len(fragments_data))

    for i in range(len(fragments_data)):
        # If we went through all the fragments in the solution,
        # all the fragments that are still in fragment_data are wrong
        if gd_solution_idx >= len(ground_truth_solution_data):
            wrong_frag.append(fragments_data_pixels.get(fragments_data[i][0]))
            continue # Go to the next "wrong" fragment

        while fragments_data[i][0] > ground_truth_solution_data[gd_solution_idx][0]:
            gd_solution_idx += 1
            if gd_solution_idx >= len(ground_truth_solution_data):
                continue

        # If the two fragments have the same ID, check if it is well-placed (more or less delta)
        if fragments_data[i][0] == ground_truth_solution_data[gd_solution_idx][0]:  # If the two ids are the same
            frag_in += 1
            output_normalized_angle = fragments_data[i][3]
            # The solution angle is taken in the opposite direction and need to be normalized in [-180 ; 180]
            solution_angle = ground_truth_solution_data[gd_solution_idx][3]
            solution_normalized_angle = - (solution_angle + 180) % 360 - 180

            print("===================output_normalized_angle : ", output_normalized_angle)
            print("fragments_data[i][1] : ", fragments_data[i][1])
            print("fragments_data[i][2] : ", fragments_data[i][2])
            print("solution_normalized_angle : ", solution_normalized_angle)
            print("ground_truth_solution_data[gd_solution_idx][1] : ", ground_truth_solution_data[gd_solution_idx][1])
            print("ground_truth_solution_data[gd_solution_idx][2] : ", ground_truth_solution_data[gd_solution_idx][2])

            if ((abs(fragments_data[i][1] - ground_truth_solution_data[gd_solution_idx][1])) < DELTA_X and
                    (abs(fragments_data[i][2] - ground_truth_solution_data[gd_solution_idx][2])) < DELTA_Y and
                    (abs(output_normalized_angle - solution_normalized_angle)) < DELTA_ANGLE):
                frag_well_placed.append(ground_truth_solution_pixels.get(ground_truth_solution_data[gd_solution_idx][0]))

            gd_solution_idx += 1

        elif fragments_data[i][0] < ground_truth_solution_data[gd_solution_idx][0]:
            wrong_frag.append(fragments_data_pixels.get(fragments_data[i][0]))

    print("frag_in : ", frag_in)
    print("len(frag_well_placed) : ", len(frag_well_placed))
    print("len(wrong_frag) : ", len(wrong_frag))
    # Compute the precision
    precision = (np.sum(frag_well_placed) - np.sum(wrong_frag)) / (sum(ground_truth_solution_pixels.values()))

    return precision


def evaluate_solution(fragments_output_path, solution_path, fragment_directory, DELTA_X, DELTA_Y, DELTA_ANGLE):
    print("\nThe fragments in solution.txt must be sorted by their IDs in ascending order.\n")
    fragments_data = load_fragments(fragments_output_path)
    solution_data = load_fragments(solution_path)

    precision = compute_solution_precision(fragments_data, solution_data, fragment_directory, DELTA_X, DELTA_Y, DELTA_ANGLE)
    print(f"The precision of : {fragments_output_path} is {precision * 100:.2f}%")


def sort_csv_by_first_column(input_file, output_file):
    def extract_first_value(value):
        # Split by spaces and take the first value
        value = value.split()[0]  # Get the first number in the string
        try:
            return int(value)
        except ValueError:
            return None  # If there's no valid integer, return None

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile, delimiter=' ')  # Read with space as delimiter
        rows = []

        for row in reader:
            first_value = extract_first_value(row[0])  # Clean and extract the first value from the first column
            if first_value is not None:
                row[0] = first_value  # Replace the first column with the extracted value
                rows.append(row)
            else:
                print(f"Skipping row with invalid first column value: {row[0]}")

        # Sort rows based on the cleaned first column values
        sorted_rows = sorted(rows, key=lambda row: row[0])

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=' ')  # Use space as delimiter for output
        writer.writerows(sorted_rows)  # Write sorted rows

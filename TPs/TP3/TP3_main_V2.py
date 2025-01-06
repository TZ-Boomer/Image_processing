import cv2
import numpy as np
import paths
import math
import os

FRAGMENT_DATA_PATH = paths.FRAGMENT_DATA_PATH
FRAGMENT_DIRECTORY = paths.FRAGMENT_DIRECTORY
TARGET_IMAGE_PATH = paths.TARGET_IMAGE_PATH
SOLUTION_PATH = paths.SOLUTION_PATH
program_output = "TP3_output.txt"

detector_n_features = 5000 # No more improvements when above 5000
key_points_matching_ratio = 0.6 # Lower is more restrictive

# Parameters for the solution file evaluation
DELTA_X = 100
DELTA_Y = 100
DELTA_ANGLE = 100


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

    for frag_data in fragments:
        frag_index = frag_data[0]
        image_name = f"frag_eroded_{frag_index}.png"
        image_path = os.path.join(images_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load the image : {image_name}.")
        images.append((frag_index, image))

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


def detect_and_compute(image, detector_type="SIFT"):
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
        index_params = dict(algorithm=1, trees=5)  # FLANN with KDTree
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method == "BF":
        # For SIFT : NORM_L2
        # For ORB  : NORM_HAMMING
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError("Unsupported matching method")

    try:
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error as e:
        print(f"OpenCV error in knnMatch : {e}")
        return []

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


# -----------------------------------------------------------------------------
# NEW: Partial affine approach with no scale
# -----------------------------------------------------------------------------
def ransac_affine_no_scale(kp_src, kp_dst, matches, reproj_thresh=5.0):
    """
    Use an affine model (with RANSAC) to filter matches,
    then remove any scaling by forcing scale=1 in the final matrix.

    Returns:
        M_no_scale: 2x3 matrix (rotation+translation only)
        inlier_mask: list of 0/1 for each match (inliers vs outliers)
    """
    # For affine transform, we need at least 3 matches (preferably non-collinear).
    if len(matches) < 3:
        raise ValueError("Not enough matches for an affine transform.")

    # Collect matched keypoints
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate partial 2D affine: rotation, uniform scale, translation (no shear) with RANSAC
    M_estimated, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh
    )
    if M_estimated is None:
        raise ValueError("Could not estimate partial affine transform.")

    inlier_mask = inliers.ravel().tolist()  # Nx1 -> list of 0/1

    # M_estimated is [ [a, b, tx],
    #                  [c, d, ty] ]
    # with (a,d) ~ scale*cos(theta), (b,c) ~ scale*sin(theta)

    # (1) Extract rotation angle
    angle = math.atan2(M_estimated[1, 0], M_estimated[0, 0])  # atan2(c, a)

    # (2) Force scale = 1
    cosA = math.cos(angle)
    sinA = math.sin(angle)
    tx = M_estimated[0, 2]
    ty = M_estimated[1, 2]

    # Rebuild an affine transform with no scale
    M_no_scale = np.array([
        [cosA, -sinA, tx],
        [sinA,  cosA, ty]
    ], dtype=np.float32)

    return M_no_scale, inlier_mask


def decompose_affine_no_scale(M):
    """
    Given a 2x3 matrix M = [ [cosθ, -sinθ, tx],
                             [sinθ,  cosθ, ty] ],
    extract (posx, posy, angle in degrees).
    """
    angle = math.atan2(M[1, 0], M[0, 0])  # (c, a)
    posx = M[0, 2]
    posy = M[1, 2]
    angle_deg = math.degrees(angle)
    return posx, posy, angle_deg


def overlay_fragment_on_painting_no_scale(painting, fragment, M_affine):
    """
    Warp 'fragment' onto 'painting' using warpAffine, ignoring scale.
    painting: base image
    fragment: fragment to overlay
    M_affine: 2x3 matrix with no scale (rotation + translation)
    """
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
# -----------------------------------------------------------------------------


def draw_matches(img1, kp1, img2, kp2, matches, matches_mask):
    if img1 is None or img2 is None:
        return None

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    return result


# NOTE: We keep this old homography-based function in case you need it,
# but it is now unused since we do partial affine below.
def ransac_filter(kp1, kp2, matches, reproj_thresh=5.0):
    if len(matches) < 4:
        raise ValueError("Not enough matches for RANSAC")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None:
        raise ValueError("Homography not found")

    matches_mask = mask.ravel().tolist()
    return H, matches_mask


# NOTE: We keep the old homography decomposition function if needed.
def decompose_homography(H):
    # Extract the rotation/translation part
    r00, r01, t0 = H[0, 0], H[0, 1], H[0, 2]
    r10, r11, t1 = H[1, 0], H[1, 1], H[1, 2]

    # Calculating the angle from the rotation matrix
    angle = math.degrees(math.atan2(r10, r00))

    # Positions (translation)
    posx, posy = t0, t1
    return posx, posy, angle


def get_pixels_count(data, fragment_directory):
    fragment_pixels = {}

    for fragment in data:
        path = fragment_directory + f"frag_eroded_{fragment[0]}.png"
        image = cv2.imread(path, -1) # Load with alpha channel "-1"
        alpha_channel = image[:, :, 3]
        non_transparent_pixels = np.count_nonzero(alpha_channel > 0)
        fragment_pixels[fragment[0]] = non_transparent_pixels

    return fragment_pixels


def compute_solution_precision(fragments_data, ground_truth_solution_data, fragment_directory):
    well_placed = [] # Well-placed fragments
    wrong_fragments = [] # Fragments that shouldn't be placed
    gd_solution_idx = 0

    # Get the number of pixels for each fragment in the data files
    ground_truth_solution_pixels = get_pixels_count(ground_truth_solution_data, fragment_directory)
    fragments_data_pixels = get_pixels_count(fragments_data, fragment_directory)

    for i in range(len(fragments_data)):
        #print("fragments_data[i][0] : ", fragments_data[i][0])
        #print("ground_truth_solution_data[solution_index][0] : ", ground_truth_solution_data[gd_solution_idx][0])

        # If we went through all the fragments in the solution,
        # all the fragments that are still in fragment_data are wrong
        if gd_solution_idx >= len(ground_truth_solution_data):
            wrong_fragments.append(fragments_data_pixels.get(fragments_data[i][0]))
            continue # Go to the next "wrong" fragment

        while fragments_data[i][0] > ground_truth_solution_data[gd_solution_idx][0]:
            gd_solution_idx += 1
            if gd_solution_idx >= len(ground_truth_solution_data):
                continue

        # If the two fragments have the same ID, check if it is well-placed (more or less delta)
        if fragments_data[i][0] == ground_truth_solution_data[gd_solution_idx][0]:  # If the two ids are the same
            print("fragments_data[i][0] : ", fragments_data[i][0])
            print("ground_truth_solution_data[solution_index][0] : ", ground_truth_solution_data[gd_solution_idx][0])

            if ((abs(fragments_data[i][1] - ground_truth_solution_data[gd_solution_idx][1])) < DELTA_X and
                    (abs(fragments_data[i][2] - ground_truth_solution_data[gd_solution_idx][2])) < DELTA_Y and
                    (abs(fragments_data[i][3] - ground_truth_solution_data[gd_solution_idx][3])) < DELTA_ANGLE):
                well_placed.append(ground_truth_solution_pixels.get(ground_truth_solution_data[gd_solution_idx][0]))

            gd_solution_idx += 1

        elif fragments_data[i][0] < ground_truth_solution_data[gd_solution_idx][0]:
            wrong_fragments.append(fragments_data_pixels.get(fragments_data[i][0]))

    print("ground_truth_solution_pixels.values() : ", ground_truth_solution_pixels.values())
    # Compute the precision
    precision = (np.sum(well_placed) - np.sum(wrong_fragments)) / (sum(ground_truth_solution_pixels.values()))

    return precision


def evaluate_solution(fragments_output_path, solution_path, fragment_directory):
    print("\nThe fragments in solution.txt must be sorted by their IDs in ascending order.\n")
    fragments_data = load_fragments(fragments_output_path)
    solution_data = load_fragments(solution_path)

    print(f"The precision of : {fragments_output_path}")
    print(f"is : {compute_solution_precision(fragments_data, solution_data, fragment_directory) * 100:.4f}%")


def image_reconstruction(fragment_path, fragment_directory, final_image_path):
    """
    Main function for fresco reconstruction using PARTIAL AFFINE with NO SCALE:
    1) Load fragment info + images
    2) Detect & match keypoints
    3) RANSAC for partial affine (no scale)
    4) Save <index> <posx> <posy> <angle> in solutions file
    5) Optionally, overlay fragments in a black image for final reconstruction
    """
    print("----- Image Reconstruction -----")
    # Loading fragments
    fragments_data = load_fragments(fragment_path)
    fragments_images = load_images(fragments_data, fragment_directory)

    # Loading of fresco
    painting = get_painting(final_image_path, black=False)  # for matching
    reconstruction = get_painting(final_image_path, black=False)  # for overlay

    # Keypoints & descriptors of the fresco
    kp_painting, desc_painting = detect_and_compute(painting, "SIFT")

    # Create solution file
    f_out = open(program_output, 'w')

    n_frag_placed = 0

    for frag_index, frag_img in fragments_images:
        print(f"--- Processing fragment {frag_index} ---")

        # Key points & descriptors of the fragment
        kp_frag, desc_frag = detect_and_compute(frag_img, "SIFT")

        # Matching
        matches = match_keypoints(desc_frag, desc_painting, method="FLANN", ratio_thresh=key_points_matching_ratio)
        print(f"Number of matches before RANSAC : {len(matches)}")

        if len(matches) < 3:
            # For partial affine, we need at least 3 matches (without rescaling)
            print(f"Not enough good matches (>=3) for fragment {frag_index}.")
            continue

        # Estimate partial affine with RANSAC, no scale
        try:
            M_no_scale, inlier_mask = ransac_affine_no_scale(kp_frag, kp_painting, matches, reproj_thresh=5.0)
        except ValueError as e:
            print(f"Error computing partial affine for fragment {frag_index}: {e}")
            continue

        # Decompose to get (posx, posy, angle)
        posx, posy, angle = decompose_affine_no_scale(M_no_scale)

        # Write to solutions file
        f_out.write(f"{frag_index} {int(posx)} {int(posy)} {angle:.2f}\n")
        print(f"Fragment {frag_index} => posx={int(posx)}, posy={int(posy)}, angle={angle:.2f}")

        # Visual overlay using affine warp
        reconstruction = overlay_fragment_on_painting_no_scale(reconstruction, frag_img, M_no_scale)
        n_frag_placed += 1

        # (Optional) debug match visualization
        # matches_img = draw_matches(frag_img, kp_frag, painting, kp_painting, matches, inlier_mask)
        # if matches_img is not None:
        #     cv2.imshow(f"Matches Fragment {frag_index}", matches_img)
        #     cv2.waitKey(500)

    f_out.close()
    print(f"\nGenerated solutions file : {program_output}")

    # Final result
    cv2.imwrite("reconstruction_result.png", reconstruction)
    print("Final reconstruction saved : reconstruction_result.png")

    print(f"Number of fragments placed on the fresco : {n_frag_placed}.")

    show_image(reconstruction)


# Example usage
def main():
    #image_reconstruction(FRAGMENT_DATA_PATH, FRAGMENT_DIRECTORY, TARGET_IMAGE_PATH)
    evaluate_solution(program_output, SOLUTION_PATH, FRAGMENT_DIRECTORY)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import paths
import math
import os


FRAGMENT_PATH = paths.FRAGMENT_PATH
FRAGMENT_DIRECTORY = paths.FRAGMENT_DIRECTORY
TARGET_IMAGE_PATH = paths.TARGET_IMAGE_PATH


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
        detector = cv2.SIFT_create()
    elif detector_type == "ORB":
        detector = cv2.ORB_create()
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
        # Pour SIFT : NORM_L2
        # Pour ORB  : NORM_HAMMING
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


def decompose_homography(H):
    # Extract the rotation/translation part
    r00, r01, t0 = H[0, 0], H[0, 1], H[0, 2]
    r10, r11, t1 = H[1, 0], H[1, 1], H[1, 2]

    # Calculating the angle from the rotation matrix
    angle = math.degrees(math.atan2(r10, r00))

    # Positions (translation)
    posx, posy = t0, t1

    return posx, posy, angle


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


def overlay_fragment_on_painting(painting, fragment, H):
    if fragment is None or painting is None:
        return painting

    h, w, _ = painting.shape
    warped_fragment = cv2.warpPerspective(fragment, H, (w, h))

    # To embed we assume that the background of the fragment is black or transparent.
    # If it is black, we can make a simple mask.
    fragment_gray = cv2.cvtColor(warped_fragment, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fragment_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Mask the corresponding area in the painting
    painting_bg = cv2.bitwise_and(painting, painting, mask=mask_inv)
    # Combines fragment and painting
    painting_fg = cv2.bitwise_and(warped_fragment, warped_fragment, mask=mask)
    final = cv2.add(painting_bg, painting_fg)

    return final


def image_reconstruction(fragment_path, fragment_directory, final_image_path):
    """
    Fonction principale pour la reconstruction de la fresque :
    1) Chargement des infos (index, etc.) + images
    2) Détection et matching des points d'intérêt
    3) RANSAC, homographie
    4) Sauvegarde <index> <posx> <posy> <angle> dans un fichier de solutions
    5) (Optionnel) Incrustation visuelle des fragments
    """
    print("----- Image Reconstruction -----")
    # Loading fragments
    fragments_data = load_fragments(fragment_path)
    fragments_images = load_images(fragments_data, fragment_directory)

    # Loading of fresco
    painting = get_painting(final_image_path, black=False) # For matching
    reconstruction = get_painting(final_image_path, black=True) # For overlay

    # Key points and description of fresco
    kp_painting, desc_painting = detect_and_compute(painting, "SIFT")

    # Creating solution file
    solutions_file = "solutions.txt"
    f_out = open(solutions_file, 'w')

    for frag_index, frag_img in fragments_images:
        print(f"--- Processing fragment {frag_index} ---")

        # Key points and description
        kp_frag, desc_frag = detect_and_compute(frag_img, "SIFT")

        # Matching between the fragment and the fresco
        matches = match_keypoints(desc_frag, desc_painting, method="FLANN", ratio_thresh=0.7)
        print(f"Number of matches before RANSAC : {len(matches)}")

        if len(matches) < 4:
            print(f"Not enough good matches for fragment {frag_index}.")
            continue

        # RANSAC to estimate the homography
        try:
            H, mask = ransac_filter(kp_frag, kp_painting, matches, reproj_thresh=5.0)
        except ValueError as e:
            print(f"Error RANSAC for fragment {frag_index} : {e}")
            continue

        # Decomposition to obtain (posx, posy, angle)
        posx, posy, angle = decompose_homography(H)

        # Writing to the solutions file
        f_out.write(f"{frag_index} {int(posx)} {int(posy)} {angle:.2f}\n")
        print(f"Fragment {frag_index} => posx={int(posx)}, posy={int(posy)}, angle={angle:.2f}")

        # Visual overlay
        reconstruction = overlay_fragment_on_painting(reconstruction, frag_img, H)

        # (Optionnel) Visualisation des matches pour debug
        # matches_img = draw_matches(frag_img, kp_frag, painting, kp_painting, matches, mask)
        # if matches_img is not None:
        #     cv2.imshow(f"Matches Fragment {frag_index}", matches_img)
        #     cv2.waitKey(500)

    f_out.close()
    print(f"Generated solutions file : {solutions_file}")

    # Final result
    cv2.imwrite("reconstruction_result.png", reconstruction)
    print("Final reconstruction saved : reconstruction_result.png")

    show_image(reconstruction)


# Example usage
def main():
    image_reconstruction(FRAGMENT_PATH, FRAGMENT_DIRECTORY, TARGET_IMAGE_PATH)    # Load fragment images


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import paths
import math
import TP3_tools


FRAGMENT_DATA_PATH = paths.FRAGMENT_DATA_PATH
FRAGMENT_DIRECTORY = paths.FRAGMENT_DIRECTORY
TARGET_IMAGE_PATH = paths.TARGET_IMAGE_PATH
SOLUTION_PATH = paths.SOLUTION_PATH
program_output = "TP3_main_output.txt"

detector_n_features = 5000 # No more improvements when above 5000
key_points_matching_ratio = 0.6 # Lower is more restrictive
BLACK_FRESCO = True

# Parameters for the solution file evaluation
DELTA_X = 10
DELTA_Y = 10
DELTA_ANGLE = 10


def ransac_affine_no_scale(kp_frag, kp_fresco, matches, reproj_thresh=5.0):
    # For affine transform, we need at least 3 matches (non-collinear).
    if len(matches) < 3:
        raise ValueError("Not enough matches for an affine transform.")

    # Collect matched keypoints
    frag_pts = np.float32([kp_frag[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    fresco_pts = np.float32([kp_fresco[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate partial 2D affine: rotation, uniform scale, translation (no shear) with RANSAC
    M_estimated, inliers = cv2.estimateAffinePartial2D(
        frag_pts, fresco_pts,
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


def image_reconstruction(fragment_path, fragment_directory, target_image_path):
    print("----- Image Reconstruction -----")
    # Loading fragments
    fragments_data = TP3_tools.load_fragments(fragment_path)
    fragments_images = TP3_tools.load_images(fragments_data, fragment_directory)

    # Loading of fresco
    painting = TP3_tools.get_painting(target_image_path, black=False)  # for matching
    reconstruction = TP3_tools.get_painting(target_image_path, black=BLACK_FRESCO)  # for overlay

    # Keypoints & descriptors of the fresco
    kp_painting, desc_painting = TP3_tools.detect_and_compute(painting, detector_n_features, "SIFT")

    # Create solution file
    f_out = open(program_output, 'w')
    # Count the number of fragment placed
    n_frag_placed = 0

    for frag_index, frag_img in fragments_images:
        print(f"--- Processing fragment {frag_index} ---")

        # Key points & descriptors of the fragment
        kp_frag, desc_frag = TP3_tools.detect_and_compute(frag_img, detector_n_features, "SIFT")

        # Matching
        matches = TP3_tools.match_keypoints(desc_frag, desc_painting, method="FLANN", ratio_thresh=key_points_matching_ratio)
        print(f"Number of matches before RANSAC : {len(matches)}")

        if len(matches) < 3:
            # For partial affine, we need at least 3 matches
            print(f"Not enough good matches (>=3) for fragment {frag_index}.")
            continue

        # Estimate partial affine with RANSAC, no scale
        try:
            M_affine, inlier_mask = ransac_affine_no_scale(kp_frag, kp_painting, matches, reproj_thresh=5.0)
        except ValueError as e:
            print(f"Error computing partial affine for fragment {frag_index}: {e}")
            continue

        # Decompose to get (posx, posy, angle)
        posx, posy, angle = TP3_tools.decompose_affine_no_scale(M_affine)

        # Write to solutions file
        f_out.write(f"{frag_index} {int(posx)} {int(posy)} {angle:.2f}\n")
        print(f"Fragment {frag_index} => posx={int(posx)}, posy={int(posy)}, angle={angle:.2f}")

        # Visual overlay using affine warp
        reconstruction = TP3_tools.overlay_fragment_on_painting_no_scale(reconstruction, frag_img, M_affine)
        n_frag_placed += 1

    f_out.close()
    print(f"\nGenerated solutions file : {program_output}")

    # Final result
    cv2.imwrite("reconstruction_result.png", reconstruction)
    print("Final reconstruction saved : reconstruction_result.png")

    print(f"Number of fragments placed on the fresco : {n_frag_placed}.")

    TP3_tools.show_image(reconstruction)


# Example usage
def main():
    image_reconstruction(FRAGMENT_DATA_PATH, FRAGMENT_DIRECTORY, TARGET_IMAGE_PATH)
    TP3_tools.evaluate_solution(program_output, SOLUTION_PATH, FRAGMENT_DIRECTORY, DELTA_X, DELTA_Y, DELTA_ANGLE)


if __name__ == "__main__":
    main()

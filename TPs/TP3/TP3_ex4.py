import cv2
import numpy as np
import paths
import TP3_tools


FRAGMENT_DIRECTORY = paths.FRAGMENT_DIRECTORY
TARGET_IMAGE_PATH = paths.TARGET_IMAGE_PATH
SOLUTION_PATH = paths.SOLUTION_PATH
program_output = "TP3_ex4_output.txt"

# Distance threshold parameters
DISTANCE_THRESHOLD = 3
detector_n_features = 5000 # No more improvements when above 5000
BLACK_FRESCO = True

# Parameters to compute the precision
DELTA_X = 3
DELTA_Y = 3
DELTA_ANGLE = 2


def filter_by_distance_conservation(kp_frag, kp_fresco, matches):
    if len(matches) < 2:
        return []

    consistent_matches = []

    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            m1, m2 = matches[i], matches[j]
            frag_dist = np.linalg.norm(np.array(kp_frag[m1.queryIdx].pt) - np.array(kp_frag[m2.queryIdx].pt))
            fresco_dist = np.linalg.norm(np.array(kp_fresco[m1.trainIdx].pt) - np.array(kp_fresco[m2.trainIdx].pt))

            if abs(frag_dist - fresco_dist) <= DISTANCE_THRESHOLD:
                consistent_matches.append(m1)
                consistent_matches.append(m2)

    return list(set(consistent_matches))


def compute_geometric_transformation(kp_frag, kp_fresco, filtered_matches, reproj_thresh):
    # Extract coordinates
    frag_pts = np.float32([kp_frag[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
    fresco_pts = np.float32([kp_fresco[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

    M_affine, _ = cv2.estimateAffinePartial2D(frag_pts, fresco_pts, method=cv2.LMEDS)

    if M_affine is None:
        raise ValueError("Could not estimate partial affine transform.")

    return M_affine


def reconstruct_image(fragment_directory, target_image_path):
    fragments_images = TP3_tools.load_images(fragment_directory)
    fresco_img = cv2.imread(target_image_path)
    kp_fresco, desc_fresco = TP3_tools.detect_and_compute(fresco_img, detector_n_features)

    reconstruction = TP3_tools.get_painting(target_image_path, black=BLACK_FRESCO)

    # Create solution file
    f_out = open(program_output, 'w')

    # Count the number of fragment placed
    n_frag_placed = 0

    for frag_index, frag_img in fragments_images:
        kp_frag, desc_frag = TP3_tools.detect_and_compute(frag_img, detector_n_features)
        matches = TP3_tools.match_keypoints(desc_frag, desc_fresco)

        filtered_matches = filter_by_distance_conservation(kp_frag, kp_fresco, matches)

        if len(filtered_matches) < 3:
            print(f"Not enough consistent matches for fragment {frag_index}.")
            continue

        # Estimate partial affine with RANSAC, no scale
        try:
            M_affine = compute_geometric_transformation(kp_frag, kp_fresco, filtered_matches, reproj_thresh=5.0)
        except ValueError as e:
            print(f"Error computing partial affine for fragment {frag_index}: {e}")
            continue

        posx, posy, angle = TP3_tools.compute_frag_coordinates(fresco_img, frag_img, M_affine)

        # Write to solutions file
        f_out.write(f"{frag_index} {int(posx)} {int(posy)} {angle:.2f}\n")
        print(f"Fragment {frag_index} => posx={int(posx)}, posy={int(posy)}, angle={angle:.2f}")

        # Visual overlay using affine warp
        reconstruction = TP3_tools.overlay_fragment_on_painting_no_scale(reconstruction, frag_img, M_affine)
        n_frag_placed += 1

        print(f"Fragment {frag_index} processed successfully.")

    f_out.close()
    TP3_tools.sort_csv_by_first_column(program_output, program_output)
    print(f"\nGenerated solutions file : {program_output}")

    # Final result
    cv2.imwrite("reconstruction_result.png", reconstruction)
    print("Final reconstruction saved : reconstruction_result.png")

    print(f"Number of fragments placed on the fresco : {n_frag_placed}.")

    TP3_tools.show_image(reconstruction)


def main():
    reconstruct_image(FRAGMENT_DIRECTORY, TARGET_IMAGE_PATH)
    TP3_tools.evaluate_solution(program_output, SOLUTION_PATH, FRAGMENT_DIRECTORY, DELTA_X, DELTA_Y, DELTA_ANGLE)


if __name__ == "__main__":
    main()

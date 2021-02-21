import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from imageio import imsave
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
import sol4_utils
from scipy.ndimage.filters import convolve


RADIUS = 3

K = 0.04

X_DER = np.array([[1, 0, -1]]).astype("float64")

Y_DER = X_DER.T


def dis(im1):
    plt.figure()
    plt.imshow(im1, cmap = 'gray')
    plt.show()

# pt 3.1


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing a gray scale image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    if len(im.shape) > 2:
        return
    # getting derivatives etc.
    dx, dy = convolve(im, X_DER), convolve(im, Y_DER)
    dx_2, dy_2, dxy = np.multiply(dx, dx), np.multiply(dy, dy), np.multiply(dx, dy)
    dx_2, dy_2, dxy = sol4_utils.blur_spatial(dx_2, 3), sol4_utils.blur_spatial(dy_2, 3), sol4_utils.blur_spatial(dxy, 3)
    # computing response
    r = (np.multiply(dx_2, dy_2) - np.multiply(dxy, dxy)) - K * ((dx_2 + dy_2) ** 2)
    arr = np.argwhere(non_maximum_suppression(r))
    fixed_arr = np.empty(arr.shape)
    fixed_arr[:, 1], fixed_arr[:, 0] = arr[:, 0], arr[:, 1]
    return fixed_arr


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    # empty descriptor
    arr = np.ndarray((len(pos), 1 + (2 * desc_rad), 1 + (2 * desc_rad)))
    # normalizing
    pos = 0.25 * pos
    # calculating descriptors
    for i, pixel in enumerate(pos):
        x = np.arange(pixel[0] - desc_rad, pixel[0] + desc_rad + 1)
        y = np.arange(pixel[1] - desc_rad, pixel[1] + desc_rad + 1)
        x, y = np.meshgrid(x, y)
        d = map_coordinates(im, [y, x], order = 1, prefilter = False)
        mean = d.mean()
        if np.linalg.norm(d - mean):
            arr[i, :, :] = ((d - mean) / np.linalg.norm(d - mean))
    return arr


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    points = spread_out_corners(pyr[0], 7, 7, 12)
    return [points, sample_descriptor(pyr[2], points, RADIUS)]


# pt 3.2


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # put each desc in a row vector in the vector_desc_mat
    desc_mat1 = np.ndarray.reshape(desc1, (len(desc1), desc1.shape[1] ** 2))
    desc_mat2 = np.ndarray.reshape(desc2, (len(desc2), desc2.shape[1] ** 2))
    # multiply matrix mult- desc_mat1 @ desc_mat2.T for the scores matrix
    scores = desc_mat1 @ desc_mat2.T
    # get second maximum
    second_best_col, second_best_row = np.partition(scores, -2, axis = 0), np.partition(scores, -2, axis = 1)
    # make boolean
    second_best_row_bool = scores.T > second_best_row[:, -2]
    second_best_col_bool = scores > second_best_col[-2]
    minimum = (scores > min_score)
    arr = np.where(minimum & second_best_row_bool.T & second_best_col_bool)
    return [arr[0], arr[1]]


# pt 3.3


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12
    """
    # append 3rd dimension of 1's to pos1
    norm_pos = np.ndarray((pos1.shape[0], 3), dtype = int)
    # assign 1 to new dimension
    norm_pos[:, 2] = 1
    norm_pos[:, 0], norm_pos[:, 1] = pos1[:, 0], pos1[:, 1]
    # calculate the transformed points
    trans_pos = H12 @ norm_pos.T
    return (trans_pos[0:2, ] / trans_pos[2:3, ]).T


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only = False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    largest_set, largest_indices = [], []
    if not len(points1) and not len(points2):
        return [np.eye(3), []]
    for i in range(num_iter):
        # get a random set of points
        rand_index1, rand_index2 = np.random.choice(len(points1), 2)
        rand_points1, rand_points2 = np.array([points1[rand_index1]]), np.array([points2[rand_index1]])
        if not translation_only:
            rand_points1 = np.array([rand_points1[0], points1[rand_index2]])
            rand_points2 = np.array([rand_points2[0], points2[rand_index2]])
        # compute homography H12 and call apply_homography with it on points1
        trans_p1 = apply_homography(points1, estimate_rigid_transform(rand_points1, rand_points2, translation_only))
        inliers_set, indices = [], []
        # compute inliers and outliers
        for j in range(len(points1)):
            if (np.linalg.norm(trans_p1[j] - points2[j]) ** 2) < inlier_tol:
                inliers_set.append([points1[j], points2[j]])
                indices.append(j)
        if len(inliers_set) > len(largest_set):
            largest_indices = copy.deepcopy(indices)
            largest_set = copy.deepcopy(inliers_set)
    largest_set, largest_indices = np.array(largest_set), np.array(largest_indices)
    p1 = np.reshape(largest_set[:, :1, :], (largest_set.shape[0], largest_set.shape[2]))
    p2 = np.reshape(largest_set[:, 1:, :], (largest_set.shape[0], largest_set.shape[2]))
    return [estimate_rigid_transform(p1, p2, translation_only), largest_indices]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A gray scale image.
    :param im2: A gray scale image.
    :param points1: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    height = max(im1.shape[0], im2.shape[0])
    width = im1.shape[1] + im2.shape[1]
    points2[:, 0] = points2[:, 0] + im1.shape[1]
    im = np.ndarray((height, width))
    im[:, :im1.shape[1]] = im1
    im[:, im1.shape[1]:width] = im2
    plt.figure()
    for i in range(len(points2)):
        if i in inliers:
            plt.plot([points1[i][0], points2[i][0]], [points1[i][1], points2[i][1]], c = "y", lw = .4, ms = 5,
                     mfc = "r", marker = ".")
        else:
            plt.plot([points1[i][0], points2[i][0]], [points1[i][1], points2[i][1]], c = "b", lw = .4, ms = 5,
                     mfc = "r", marker = ".")
    plt.imshow(im, cmap = 'gray')
    plt.show()


# pt 3.4


def accumulate_homographies(h_successive, m):
    """
    Convert a list of successive homographies to a
    list of homographies to a common reference frame.
    :param h_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    hom_list = [None] * (len(h_successive) + 1)
    new_mat = np.eye(3)
    for i in reversed(range(m)):
        new_mat = h_successive[i] @ new_mat
        new_mat /= new_mat[2, 2]
        hom_list[i] = new_mat
    hom_list[m] = np.eye(3)
    new_mat = np.eye(3)
    for i in range(m, len(h_successive)):
        new_mat = np.linalg.inv(h_successive[i]) @ new_mat
        new_mat /= new_mat[2, 2]
        hom_list[i + 1] = new_mat
    return hom_list


# pt 4.1


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    corners = apply_homography(np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]), homography).astype(np.int)
    return np.array([[min(corners[:, 0]), min(corners[:, 1])], [max(corners[:, 0]), max(corners[:, 1])]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    trans_corners = compute_bounding_box(homography, image.shape[0], image.shape[1])
    # black image in the right size
    canvas = np.zeros((trans_corners[1][0] - trans_corners[0][0], trans_corners[1][1] - trans_corners[0][1]))
    x_range, y_range = np.meshgrid(np.arange(canvas.shape[0]), np.arange(canvas.shape[1]))
    trans_grid = np.array([x_range.ravel(), y_range.ravel()]).T
    temp_grid = np.copy(trans_grid)
    temp_grid[:, 0] += trans_corners[0][0]
    temp_grid[:, 1] += trans_corners[0][1]
    inv_trans_grid = apply_homography(temp_grid, np.linalg.inv(homography)).T
    canvas[trans_grid[:, 0], trans_grid[:, 1]] = map_coordinates(image, inv_trans_grid, order = 1, prefilter = False)
    return canvas

# ====================================
# ====== Built-in Functions ==========
# ====================================


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only = False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis = 0)
    centroid2 = points2.mean(axis = 0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    # else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1

    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

    h = np.eye(3)
    h[:2, :2] = rotation
    h[:2, 2] = translation
    return h


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    local_max[image<(image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:,0], centers[:,1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype = np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype = np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype = np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        self.bounding_boxes = None
        self.frames_for_panoramas = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only = False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        points_and_descriptors = []
        for i, file in enumerate(self.files):
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))
        h_s = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]
            h12, inliers = ransac_homography(points1, points2, 200, 6, translation_only)
            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            h_s.append(h12)
        accumulated_homographies = accumulate_homographies(h_s, (len(h_s) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation = 5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None
        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by
        # the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)
        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis = (0, 1))
        self.bounding_boxes -= global_offset
        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint = True, dtype = np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate
            # system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]
        panorama_size = np.max(self.bounding_boxes, axis = (0, 1)).astype(np.int) + 1
        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)
        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype = np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]
            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        # assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        for i in range(len(self.panoramas)):
            dis(self.panoramas[i])
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        for i, panorama in enumerate(self.panoramas):
            imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize = (20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize = figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

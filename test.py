
from os.path import join, dirname

from sol4 import *
from sol4_utils import *
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np


# from imageio import imwrite


def show_im(im, is_gray = False):
    if is_gray:
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    plt.show()


def relpath(*filename):
    """
    as in ex3.pdf description
    """
    return join(dirname(__file__), *filename)


def test_harris_corner_detector(im):
    assert im.ndim == 2, "input image should be gray"
    show_im(im, True)
    xy_values = harris_corner_detector(im)
    assert xy_values.shape[1] == 2
    plt.imshow(im, cmap = 'gray')
    for xy in xy_values:
        plt.scatter(xy[0], xy[1], s = 130, facecolors = 'none', edgecolors = 'r')
    plt.show()
    xy_values = spread_out_corners(im, 3, 3, 60)
    plt.imshow(im, cmap = 'gray')
    for xy in xy_values:
        plt.scatter(xy[0], xy[1], s = 130, facecolors = 'none', edgecolors = 'b')
    plt.show()


def test_sample_descriptor(im):
    assert im.ndim == 2, "input image should be gray"
    xy_values = harris_corner_detector(im)
    descriptors = sample_descriptor(im, xy_values, 3)
    assert descriptors.ndim == 3
    assert descriptors.shape[0] == xy_values.shape[0]
    assert descriptors.shape[1] == descriptors.shape[2] == 7


def test_find_features(im, specific_corner = 0):
    pyr, _ = build_gaussian_pyramid(im, 3, 3)
    corners, descriptors = find_features(pyr)
    assert descriptors.ndim == 3
    assert descriptors.shape[1] == descriptors.shape[2] == 7
    assert len(corners) == len(descriptors)
    assert np.all(corners >= np.array([3, 3]))
    assert np.all(corners < np.array([im.shape[1] - 3, im.shape[0] - 3]))

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im, 'gray')
    ax1.scatter(corners[specific_corner][0], corners[specific_corner][1], s = 130, facecolors = 'none',
                edgecolors = 'r')
    ax2.imshow(descriptors[specific_corner].transpose(), 'gray')
    plt.show()


def test_matches(im1, im2, n_lines = 10):
    pyr1, _ = build_gaussian_pyramid(im1, 3, 3)
    corners1, descriptors1 = find_features(pyr1)
    pyr2, _ = build_gaussian_pyramid(im2, 3, 3)
    corners2, descriptors2 = find_features(pyr2)
    xs, ys = match_features(descriptors1, descriptors2, 0.5)
    assert len(xs) == len(ys)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im1, 'gray')
    ax2.imshow(im2, 'gray')
    for index in np.random.choice(np.arange(len(xs)), n_lines):
        x, y = xs[index], ys[index]
        con = ConnectionPatch(corners2[y], corners1[x], coordsA = 'data', coordsB = 'data', axesA = ax2,
                              axesB = ax1, color = 'red')
        ax2.add_artist(con)

    plt.show()


def test_apply_homography():
    pos1 = np.array([[6, 5], [2, 0], [63, 57], [23, 90], [2, 5]])
    H12 = np.array([[1, 0, 5],
                    [0, 1, 5],
                    [0, 0, 1]])
    res = apply_homography(pos1, H12)
    assert np.all(pos1 + 5 == res)


def test_display_matches(im1, im2):
    pyr1, _ = build_gaussian_pyramid(im1, 3, 3)
    corners1, descriptors1 = find_features(pyr1)
    pyr2, _ = build_gaussian_pyramid(im2, 3, 3)
    corners2, descriptors2 = find_features(pyr2)
    matches = match_features(descriptors1, descriptors2, 0.9)
    points1, points2 = corners1[matches[0]], corners2[matches[1]]
    homography, inliers = ransac_homography(points1, points2, 1000, 10)
    assert homography.shape == (3, 3)
    assert homography[2, 2] == 1
    assert inliers.ndim == 1

    display_matches(im1, im2, points1, points2, inliers)


def test_accumulate_homographies():
    def eq_homos(exp, act):
        a = exp[:]
        a = exp / exp[2][2]
        b = act
        if np.allclose(a, b):
            return
        if np.isnan(a).any() and np.isnan(b).any():
            return
        assert False, "expected: \n" + str(a) + "\ngot:\n" + str(b) + '\n'

    h01 = np.array([[1, 0, 3],
                    [0, 1, 3],
                    [0, 0, 1]], dtype = np.float64)
    h12 = np.array([[1, 0, -2],
                    [0, 1, -1],
                    [0, 0, 1]], dtype = np.float64)
    h23 = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]], dtype = np.float64)
    h34 = np.array([[0, 1, 4],
                    [1, 0, 4],
                    [0, 0, 1]], dtype = np.float64)
    assert np.linalg.matrix_rank(h01) == 3
    assert np.linalg.matrix_rank(h12) == 3
    assert np.linalg.matrix_rank(h23) == 3
    assert np.linalg.matrix_rank(h34) == 3

    h02 = np.dot(h01, h12)
    h03 = np.dot(h02, h23)
    h04 = np.dot(h03, h34)
    h13 = np.dot(h12, h23)
    h24 = np.dot(h23, h34)
    h14 = np.dot(h12, h24)

    h00 = h11 = h22 = h33 = h44 = np.eye(3)
    h10 = np.linalg.inv(h01)
    h21 = np.linalg.inv(h12)
    h32 = np.linalg.inv(h23)
    h43 = np.linalg.inv(h34)
    h20 = np.linalg.inv(h02)
    h30 = np.linalg.inv(h03)
    h40 = np.linalg.inv(h04)
    h31 = np.linalg.inv(h13)
    h41 = np.linalg.inv(h14)
    h42 = np.linalg.inv(h24)
    h = np.stack([h01, h12, h23, h34])
    assert np.linalg.inv(h) is not None  # just to check that matrices are inferable
    output = accumulate_homographies(h, 0)
    eq_homos(h00, output[0])
    eq_homos(h10, output[1])
    eq_homos(h20, output[2])
    eq_homos(h30, output[3])
    eq_homos(h40, output[4])
    output = accumulate_homographies(h, 1)
    eq_homos(h01, output[0])
    eq_homos(h11, output[1])
    eq_homos(h21, output[2])
    eq_homos(h31, output[3])
    eq_homos(h41, output[4])
    output = accumulate_homographies(h, 2)
    eq_homos(h02, output[0])
    eq_homos(h12, output[1])
    eq_homos(h22, output[2])
    eq_homos(h32, output[3])
    eq_homos(h42, output[4])
    output = accumulate_homographies(h, 3)
    eq_homos(h03, output[0])
    eq_homos(h13, output[1])
    eq_homos(h23, output[2])
    eq_homos(h33, output[3])
    eq_homos(h43, output[4])

    output = accumulate_homographies(h, 4)
    eq_homos(h04, output[0])
    eq_homos(h14, output[1])
    eq_homos(h24, output[2])
    eq_homos(h34, output[3])
    eq_homos(h44, output[4])


def test_compute_bounding_box():
    a = compute_bounding_box(np.eye(3), 50, 100)
    assert a.dtype == np.int
    assert np.allclose(np.array(a), np.array([[0, 0], [50, 100]]))
    mat = np.array([[1, 0, 50],
                    [0, 0, 100],
                    [0, 0, 1]], dtype = np.float64)
    a = compute_bounding_box(mat, 50, 100)
    assert np.allclose(np.array(a), np.array([[50, 100], [100, 100]]))
    # mat = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0.01, 0.01, 1]], dtype=np.float64)
    # a = compute_bounding_box(mat, 50, 100)
    # print(a)


def test_warp_channel(im):
    # homography = np.array([[0, 1, 0],
    #                        [-1, 0, 0],
    #                        [0, 0, 1]], dtype=np.float64)
    # import math as m
    # a= m.pi/5
    # homography = np.array([[m.cos(a), -m.sin(a), 0],
    #                        [m.sin(a), m.cos(a), 0],
    #                        [0, 0, 1]], dtype=np.float64)
    homography = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0.001, 0.001, 1]], dtype = np.float64)
    warped = warp_channel(im, homography)
    show_im(warped, True)


oxford1 = read_image(relpath('external', 'oxford1.jpg'), 1)
oxford2 = read_image(relpath('external', 'oxford2.jpg'), 1)

# test_harris_corner_detector(oxford1)
# test_sample_descriptor(oxford1)
# test_find_features(oxford1, 10)
# test_matches(oxford1, oxford2)
# test_apply_homography()
# for _ in range(10):
#     test_display_matches(oxford1, oxford2)

test_accumulate_homographies()
# test_compute_bounding_box()

# test_warp_channel(oxford1)

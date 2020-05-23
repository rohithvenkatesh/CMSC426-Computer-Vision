import math

import numpy as np
from PIL import Image


def estimate_transformation(pointsa, pointsb):
    # This function computes a transformation that maps the locations in pointsb
    # to the locations in pointsa using least squares
    matrix = []
    for p1, p2 in zip(pointsa, pointsb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pointsb).reshape(2*len(pointsa))

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def compute_error(pointsa, transformation, pointsb):
    # transform all point in pointsb according to transformation, and compute
    # the total error made by the transformation

    m = [[transformation[0], transformation[1], transformation[2]],
         [transformation[3], transformation[4], transformation[5]],
         [transformation[6], transformation[7], 1]]
    T = np.matrix(m)
    T = np.linalg.inv(T)

    transb = []
    for (x, y) in pointsb:
        p = np.matrix([[x], [y], [1]])
        tp = np.matmul(T, p)
        tpo = (tp[0, 0]/tp[2, 0], tp[1, 0]/tp[2, 0])
        transb.append(tpo)

    err = 0
    for p1, p2 in zip(pointsa, transb):
        err = err + math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    return err/len(pointsb)


def main():
    im = Image.open('../halfdome-07.png')
    w, h = im.size
    locs1 = [(100, 100), (200, 100), (100, 200), (200, 200), (150, 150)]
    locs2 = [(0, 0), (w, 0), (0, h), (w, h), (400, 400)]
    coeffs = estimate_transformation(locs1, locs2)
    im.transform((300, 300), Image.PERSPECTIVE, coeffs, Image.BILINEAR).save('output.png')

    print(compute_error([locs1[3]], coeffs, [locs2[3]]))
    print(compute_error(locs1, coeffs, locs2))


if __name__ == "__main__":
    main()

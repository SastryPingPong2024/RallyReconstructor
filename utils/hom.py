import numpy as np
import cv2

import matplotlib.pyplot as plt

def find_intersection(p0, p1, p2, p3):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    a1 = y0 - y1
    b1 = x1 - x0
    c1 = (x0-x1)*y0 + (y1-y0)*x0
    a2 = y2 - y3
    b2 = x3 - x2
    c2 = (x2-x3)*y2 + (y3-y2)*x2
    intersection = [
        (b1*c2-b2*c1) / (a1*b2-a2*b1),
        (c1*a2-c2*a1) / (a1*b2-a2*b1)
    ]
    return intersection

def find_table_homography_matrix(table_boundary, w=900, h=500, offset=np.array([0, 0])):
    dst_pts = np.array([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h],      
    ], dtype=np.float32) + offset
    H, _ = cv2.findHomography(table_boundary, dst_pts)
    return H

def get_homography_matrix(table_boundary, table_base, table_indent, verbose=False, orig_frame=None):
    """
    Given a set of points that defines the boundary of the ping-pong table and its height above the ground,
    finds a homography relative to the set of points that defines the boundary of the table's projection onto the ground.
    """
    
    # Compute the vanishing point of the table surface
    vanishing_point = find_intersection(table_boundary[0], table_boundary[2], table_boundary[1], table_boundary[3])

    # Compute intersection points of the table base with table surface
    H = find_table_homography_matrix(table_boundary)
    H_inv = np.linalg.inv(H)

    # Frontal corners
    v1 = H_inv @ np.array([0,   500-table_indent, 1])
    v2 = H_inv @ np.array([900, 500-table_indent, 1])
    p1 = np.array([v1[0] / v1[2], table_base])
    p2 = np.array([v2[0] / v2[2], table_base])

    # Back corners
    x1 = table_boundary[0][0]
    x2 = table_boundary[1][0]
    y1 = (vanishing_point[1] - p1[1]) * (x1 - p1[0]) / (vanishing_point[0] - p1[0]) + p1[1]
    y2 = (vanishing_point[1] - p2[1]) * (x2 - p2[0]) / (vanishing_point[0] - p2[0]) + p2[1]

    # Putting it all together
    base_boundary = np.array([
        [x1, y1],
        [x2, y2],
        p1,
        p2
    ])
    
    H_base = find_table_homography_matrix(base_boundary, w=900, h=500-table_indent, offset=np.array([-450, -250]))
    
    if verbose:
        plt.imshow(orig_frame)
        plt.plot([table_boundary[2][0], vanishing_point[0]], [table_boundary[2][1], vanishing_point[1]], color="blue")
        plt.plot([table_boundary[3][0], vanishing_point[0]], [table_boundary[3][1], vanishing_point[1]], color="blue")
        plt.plot([table_boundary[0][0], table_boundary[1][0]], [table_boundary[0][1], table_boundary[1][1]], color="blue")
        plt.plot([table_boundary[2][0], table_boundary[3][0]], [table_boundary[2][1], table_boundary[3][1]], color="blue")
        plt.plot([p1[0], vanishing_point[0]], [p1[1], vanishing_point[1]], color="red")
        plt.plot([p2[0], vanishing_point[0]], [p2[1], vanishing_point[1]], color="red")
        plt.plot([x1, x2], [y1, y2], color="red")
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red")
        plt.scatter(table_boundary[:, 0], table_boundary[:, 1], color="blue")
        plt.scatter(base_boundary[:, 0], base_boundary[:, 1], color="red")
        plt.scatter([vanishing_point[0]], [vanishing_point[1]], color="purple")
        plt.show()
        plt.close()
    
    return H, H_base

def transform(pts, H):
    if pts.shape == (2,): pts = [pts]
    pts = np.array([pts])
    transformed_pts = cv2.perspectiveTransform(pts, H)[0]
    if len(transformed_pts) == 1: transformed_pts = transformed_pts[0]
    return transformed_pts
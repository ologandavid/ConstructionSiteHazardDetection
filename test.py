import numpy as np
import cv2

x = np.array([[1,2,3], [4,5, 6], [7,8,9]])


def generate_output(width, height, boundary_points, wld_coords, A):
    """
    Generates Plots of Worker Positions in Image
    """

    #Generate a Blank image
    #Transform boundarypoints, plot them on image
    #plot wrld coords on image
    out = create_blank(width, height, rgb_color=(255,255,255))
    b_width, b_height = width-100, height-100
    cv2.imshow("Workers",out)
    boundary_points= np.transpose(boundary_points)
    t_boundary = A@boundary_points
    #Find the Maximum and Minimum x,y values for the boundary Points
    a = np.amax(t_boundary, axis = 1)
    b = np.amin(t_boundary, axis = 1)
    x_max, x_min = a[0], b[0]
    y_max, y_min = a[1], b[1]
    bx, by = x_max - x_min, y_max - y_min
    # Scale by x*(300/bx), y*(300/by)
    transform_x = np.array([[300/bx],[0],[0]])
    transform_y = np.array([[0],[300/by],[0]])
    x_out = np.transform(t_boundary)*transform_x # nx1 array of x values
    y_out = np.transform(t_boundary)*transform_y # nx1 array of y values
    #Round and convert to the correct type format
    x_out = np.uint8(round(x_out))
    y_out = np.uint8(round(y_out))
    #Stack the x and y coordinates into a nX2 matrix for cv2.polylines
    pts = np.hstack(x_out, y_out)
    #Drawing Outline of Grid
    out = cv2.polylines(out, [pts], True, (0,0,0))
    for i in wrld_coord:
        #Iterating over World Coord and Plotting each Point
        #Idk the format of world coordiantes so I havent added in the indexing yet.
        #Pete probably knows better than I do.
        out = cv2.circle(out, (wrld_coord(), wrld_coord()), 5, (255,255,255), 10)
    cv2.imshow("Workers", out)
    return out

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """
    Generates a blank image, given dimensions
    """
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


width, height = 400,400
boundary_points = 
wrld_coord = 
A = 
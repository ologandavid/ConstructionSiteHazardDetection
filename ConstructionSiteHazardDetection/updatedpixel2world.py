import cv2
import numpy as np
import argparse
import pathlib
import os
import open3d as o3d
import matplotlib.pyplot as plt

"""
This script is used to define work site regions and hazards within them in order
to determine worker proximity to danger. Use an image from the user and allow 
them to select points that will define: the transformation matrix, the work site
boundary, and the hazard boundary. The image should be a frame from the video
the user wishes to analyze. The script will then read the results of YOLO
(separate script) frame by frame and apply the appropriate coordinate
transformations and produce frame by frame plots of worker positions in relation
to all boundaries.
"""

# Ask the user for the image name
#image_name = input('Filename of left image: \n')
image_name = 'Concrete_pour_site.png'
image_ext = image_name[-4:]
image_name = image_name[:-4]
path = pathlib.Path(__file__).parent.resolve()
path_img = os.path.join(path,"project-images", image_name + image_ext)
img = cv2.imread(path_img)
img_txt = cv2.imread(path_img)

h, w = img.shape[:2]

pointcount = 0
imgpoints = []
globalpoints = []
boundarypoints = []

pointcount2 = 0 
img_txt2 = cv2.imread(path_img)
imgpoints2 = []

pointcount3 = 0 
img_txt3 = cv2.imread(path_img)
hazardpoints = []

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

def generate_output(width, height, boundary_points, wld_coords, hazardpts, A):
    """
    Generates Plots of Worker Positions in Image
    Args:
        width (int)
        height (int)
        boundary_points
        wld_coords
        A: Transformation matrix 
    Ouputs:
        out (np.ndarray) Image Containing Worksite Boundary and worker positions
    """

    #Generate a Blank image
    #Transform boundarypoints, plot them on image
    #plot wrld coords on image
    buffer = 50
    out = create_blank(width, height, rgb_color=(255,255,255))
    b_width, b_height = width-100, height-100
    cv2.imshow("Workers",out)
    # Transpose to make coordinate arrays 3xN
    boundary_points= np.transpose(boundary_points)
    hazardpts = np.transpose(hazardpts)
    # Multiply to transform them into world frame
    t_boundary = A@boundary_points
    t_hazard = A@hazardpts
    # The transformation doesn't scale, ie (x,y,s) where s should =1, but doesn't
    for c in range(len(t_boundary[0])):
        t_boundary[:,c] = t_boundary[:,c]*(1/t_boundary[2,c])
    for d in range(len(t_boundary[0])):
        t_hazard[:,d] = t_hazard[:,d]*(1/t_hazard[2,d])

    #Find the Maximum and Minimum x,y values for the boundary Points
    b = np.amin(t_boundary, axis = 1)
    x_min = b[0]
    y_min = b[1]
    t_boundary[0,:] = t_boundary[0,:] - x_min
    t_boundary[1,:] = t_boundary[1,:] - y_min
    a = np.amax(t_boundary, axis = 1)

    bx, by = a[0], a[1]
    # Scale the boundary by the window's size and leave a white buffer around the edges
    transform_x = np.array([[(width-2*buffer)/bx],[0],[0]])
    transform_y = np.array([[0],[(height-2*buffer)/by],[0]])
    x_out = np.transpose(t_boundary)@transform_x+buffer # nx1 array of x values
    y_out = np.transpose(t_boundary)@transform_y+buffer # nx1 array of y values
    #Round and convert to the correct type format
    x_out = np.round(x_out)
    y_out = np.round(y_out)
    #Stack the x and y coordinates into a nX2 matrix for cv2.polylines
    pts = np.hstack((x_out, y_out))
    #Drawing Outline of Grid
    out = cv2.polylines(out, np.int32([pts]), True, (0,0,0),2)

    #Scale the hazard coordinates based on the boundary window
    x_haz = np.transpose(t_hazard)@transform_x+buffer
    y_haz = np.transpose(t_hazard)@transform_y+buffer
    #Round and convert to the correct type format
    x_haz = np.round(x_haz)
    y_haz = np.round(y_haz)
    #Stack the x and y coordinates into a nX2 matrix for cv2.polylines
    ptsh = np.hstack((x_haz, y_haz))
    #Drawing Outline of Grid
    out = cv2.polylines(out, np.int32([ptsh]), True, (0,0,255),2)


    cv2.imshow("Workers",out)
    for i in range(len(wld_coords[0])):
        #print(i)
        x_i = np.int32(np.round((width-2*buffer)/bx*(wld_coords[0][i]-x_min)))+buffer
        y_i = np.int32(np.round((height-2*buffer)/by*(wld_coords[1][i]-y_min)))+buffer
        out = cv2.circle(out, (x_i, y_i), 5, (255,0,0), 3)
    cv2.imshow("Workers", out)
    return out

def find_boundary(event, x, y, flags, params):
    """
    Selects points to establish the boundary of the image
    Inputs:
        event- user action with the mouse
        x- x position of mouse at event time
        y- y position of mouse at event time
        flags- none expected
        params- none expected
    Outputs:
        imgpoints2- Nx3 array of boundary coordinates (x,y,1)
    """
    global pointcount2
    global img_txt2
    global imgpoints2
    if(pointcount2 < 10):
        if event == cv2.EVENT_LBUTTONDOWN:
            pointcount2 = pointcount2 + 1
            print(pointcount2)
            print(x, ' ', y,' added')
            imgpoints2.append((x,y,1))
            print(imgpoints2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_txt2, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 255, 255), 3)
            cv2.imshow('image', img_txt2)
            # gx1,gy1 = input("Global coordinates of point {} x,y: ".format(pointcount2)).split(',')
            # boundarypoints.append((int(gx1),int(gy1)))
            # print(boundarypoints)

def define_hazard(event, x, y, flags, params):
    """
    Selects points to establish the boundary of a hazard
    Inputs:
        event- user action with the mouse
        x- x position of mouse at event time
        y- y position of mouse at event time
        flags- none expected
        params- none expected
    Outputs:
        hazardpoints- Nx3 array of hazard coordinates (x,y,1)
    """
    global pointcount3
    global img_txt3
    global hazardpoints
    if(pointcount3 < 10):
        if event == cv2.EVENT_LBUTTONDOWN:
            pointcount3 = pointcount3 + 1
            print(pointcount3)
            print(x, ' ', y,' added')
            hazardpoints.append((x,y,1))
            print(hazardpoints)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_txt3, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (0, 0, 255), 2)
            cv2.imshow('image', img_txt3)


def click_event(event, x, y, flags, params):
    """
    Selects 4 points to calculate the transformation matrix
    Inputs:
        event- user action with the mouse
        x- x position of mouse at event time
        y- y position of mouse at event time
        flags- none expected
        params- none expected
    Outputs:
        imgpoints- Nx2 array of coordinates (x,y)
    """
    global pointcount
    global img_txt
    global imgpoints
    if(pointcount < 4):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            pointcount = pointcount + 1
            print(pointcount)
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y,' added')
            imgpoints.append((x,y))
            print(imgpoints)
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_txt, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 3)
            cv2.imshow('image', img_txt)
            gx,gy = input("Global coordinates of point {} x,y: ".format(pointcount)).split(',')
            globalpoints.append((float(gx),float(gy)))
            print(globalpoints)

    # If a points has already been selected and the user right clicks,
    # delete the last point selected
    if(pointcount > 0):
        # checking for right mouse clicks    
        if event==cv2.EVENT_RBUTTONDOWN:
            global path_img
            pointcount = pointcount - 1
            # displaying the coordinates
            # on the Shell
            print(imgpoints[pointcount][0], ' ', imgpoints[pointcount][1]," removed.\n")
            imgpoints.pop()
            globalpoints.pop()
            #print(imgpoints)
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_txt = cv2.imread(path_img)
            for i in range(pointcount):
                oldx = imgpoints[i][0]
                oldy = imgpoints[i][1]
                cv2.putText(img_txt, str(oldx) + ',' +
                            str(oldy), (oldx,oldy), font,
                            1, (255, 0, 0), 2)
            cv2.imshow('image', img_txt)

if __name__=="__main__":

    # reading the image
    img = cv2.imread(path_img)
 
    # displaying the image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
 
    print("Select four points to calculate projection.\n")
    # setting mouse handler for the image
    # and calling the click_event() function
    #cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)

    imgpoints = np.array([[676,1015],[1632,572],[1337,493],[509,688]],np.float32)
    globalpoints = np.array([[1,0.01],[30,0.01],[30,7],[1,7]],np.float32)

    # Image point vector
    ip = np.array([imgpoints[0][0],imgpoints[0][1],0,imgpoints[1][0],imgpoints[1][1],0,imgpoints[2][0],imgpoints[2][1]])
    # A = np.array([A11,A12,A13,A21,A22,A23,A31,A32]) (8x1)
    # Gloabl point matrix
    gp = np.array([[globalpoints[0][0],globalpoints[0][1],1,0,0,0,0,0],
                   [0,0,0,globalpoints[0][0],globalpoints[0][1],1,0,0],
                   [0,0,0,0,0,0,globalpoints[0][0],globalpoints[0][1]],
                   [globalpoints[1][0],globalpoints[1][1],1,0,0,0,0,0],
                   [0,0,0,globalpoints[1][0],globalpoints[1][1],1,0,0],
                   [0,0,0,0,0,0,globalpoints[1][0],globalpoints[1][1]],
                   [globalpoints[2][0],globalpoints[2][1],1,0,0,0,0,0],
                   [0,0,0,globalpoints[2][0],globalpoints[2][1],1,0,0]])
    Aprime = np.linalg.inv(gp).dot(ip)
    Aprime = np.append(Aprime,1)
    Aprime = np.reshape(Aprime,(3,3),'C')
    A = np.linalg.inv(Aprime) # A prime takes global coords and produces image coords

    A1 = cv2.getPerspectiveTransform(imgpoints,globalpoints) # (3x3)
    #print(A)

    print("Select four points to form site boundary.\n")
    # setting mouse handler for the image
    # and calling the find_boundary() function
    cv2.setMouseCallback('image', find_boundary)
    cv2.waitKey(0)
    
    print("Select points to denote hazard.\n")
    # setting mouse handler for the image
    # and calling the find_boundary() function
    cv2.setMouseCallback('image', define_hazard)
    cv2.waitKey(0)

    # Read the text files that have the images coordinates from YOLO
    path_labels = os.path.join(path,"labels")
    filelist = os.listdir(path_labels)
    print(len(filelist))
    # YOLO outputs a file for each frame
    for i in range(len(filelist)):
        file = os.path.join(path_labels,filelist[i])
        with open(file) as f:
            line1 = [float(x) for x in next(f).split()] # read first line
            coords_txt = []
            coords_txt.append(line1)
            for line in f: # read rest of lines (a line corresponds to a different person)
                coords_txt.append([float(x) for x in line.split()])

        # [class, x, y, width, height]
        array = np.array(coords_txt)
        coords = np.transpose(array[:,1:3]) # Grab x,y coordinates only
        boxdims = np.transpose(array[:,3:5]) # width and height values
        for c in range(len(coords[0])):
            # Convert normalized coordinates to true pixel coordinates
            coords[0][c] = coords[0][c]*w
            coords[1][c] = (coords[1][c]*h)+(boxdims[1][c]*h/2)

        # Add row of 1s to make coordinates homogeneous
        homog_coords = np.append(coords,np.ones((1,len(coords[0]))),axis=0)
        # Multiply by perspective transformation matrix to convert to world frame
        wld_coords = A1@homog_coords
        # Result is not scaled properly, ie third element in vector doesn't =1
        for c in range(len(wld_coords[0])):
            wld_coords[:,c] = np.absolute(wld_coords[:,c]*(1/wld_coords[2,c]))

        # Create the plot showing: site boundary, hazard boundary, worker positions
        frame_i = generate_output(1000, 1000, imgpoints2, wld_coords, hazardpoints, A1)

        """
        Save each frame for stitching by a separate function.
        path is already defined as the parent folder of this script.
        """
        save_path = os.path.join(path,"project-frames","frame_" + str(i) + ".png")
        cv2.imwrite(save_path,frame_i)
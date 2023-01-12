import cv2
import numpy as np
import argparse
import pathlib
import os
import open3d as o3d
import matplotlib.pyplot as plt

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

def run_frames():
    """
    Cycles through frames to generate an output video
    """
    pass

def generate_output(width, height, boundary_points, wld_coords, A):
    """
    Generates Plots of Worker Positions in Image
    Args:
        width (int) : Output Image Pixel Width
        height (int) : Output Image Pixel Height
        boundary_points (), (np.ndarray), Array of Boundar0 of Worksite
        wld_coords (3xN), )(np.ndarray), Array of (x,y) Worker Positions
        A: Transformation matrix 
    Ouputs:
        out (np.ndarray) Image Containing Worksite Boundary and worker positions
    """

    #Generate a Blank image
    #Transform boundarypoints, plot them on image
    #plot wrld coords on image
    out = create_blank(width, height, rgb_color=(153,153,255))
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
    out = cv2.fillPoly(out, [pts], (153,255,204))
    for i in wld_coords:
        #Iterating over World Coord and Plotting each Point
        #Idk the format of world coordiantes so I havent added in the indexing yet.
        #Pete probably knows better than I do.
        out = cv2.circle(out, (wld_coords(), wld_coords()), 5, (255,255,255), 10)
    cv2.imshow("Workers", out)
    return out

def find_boundary(event, x, y, flags, params):
    """
    Selects 4 points to establish the boundary of the image

    """
    global pointcount2
    global img_txt2
    global imgpoints2
    if(pointcount2 < 4):
        if event == cv2.EVENT_LBUTTONDOWN:
            pointcount1 = pointcount1 + 1
            print(pointcount1)
            print(x, ' ', y,' added')
            imgpoints2.append((x,y,1))
            print(imgpoints2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_txt2, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img_txt2)
            gx1,gy1 = input("Global coordinates of point {} x,y: ".format(pointcount1)).split(',')
            boundarypoints.append((int(gx1),int(gy1)))
            print(boundarypoints)


def click_event(event, x, y, flags, params):
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
            imgpoints.append((x,y,1))
            print(imgpoints)
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_txt, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img_txt)
            gx,gy = input("Global coordinates of point {} x,y: ".format(pointcount)).split(',')
            globalpoints.append((int(gx),int(gy)))
            print(globalpoints)

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
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)

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
    #print(A)

    # point1 = imgpoints[0]
    # point1 = np.append(point1,1)
    # point1.shape = (3,1)
    # print(np.matmul(np.linalg.inv(A),point1))

    # Create a blank image to track worker positions throughout frames

    # Create new blank 300x300 white image

    # Read the text files that have the images coordinates from YOLO
    path_labels = os.path.join(path,"labels")
    filelist = os.listdir(path_labels)
    print(len(filelist))
    # YOLO outputs a file for each frame
    for i in range(len(filelist)):
        file = os.path.join(path_labels,filelist[i])
        with open(file) as f:
            line1 = [float(x) for x in next(f).split()] # read first line
            coords = []
            coords.append(line1)
            for line in f: # read rest of lines (a line corresponds to a different person)
                coords.append([float(x) for x in line.split()])

        print("File ")

        # [class, x, y, width, height]
        array = np.array(coords)
        coords = np.transpose(array[:,1:3]) # Grab x,y coordinates only
        for c in range(len(coords)):
            # Convert normalized coordinates to true pixel coordinates
            coords[0][c] = coords[0][c]*w
            coords[1][c] = coords[1][c]*h

        homog_coords = np.append(coords,np.ones((1,len(coords[0]))))
        wld_coords = A@homog_coords

        # draw a cirlce on image with wld_coords
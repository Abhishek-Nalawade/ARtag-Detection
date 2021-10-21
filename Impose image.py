from scipy import fft
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


# used to compute the norm
def magnitude(mat):
    sum = 0
    for i in mat:
        sum = sum + (i)**2
    root = sum**(1/2)
    return root


#finds the orientation of the tag and returns the corresponding index
##
    # brief finds the orientation of the tag
    # details This function finds the orientation of the tag by dividing
    #         the frame into a grid of 8 x 8
    #
    # @param AR is the homographied image of the ARtag
    # return int returns index of corresponding orientation
##
def orientation(AR):
    img = cv2.cvtColor(AR, cv2.COLOR_BGR2GRAY)
    img[img[:,:]>180] = 255
    img[img[:,:]<180] = 0
    cv2.imshow("from_function",img)
    width = 160
    height = 160
    coor = [[3,3],[6,3],[6,6],[3,6]]
    gridx = int(width/8)
    gridy = int(height/8)
    reverse_code = list()
    b = cv2.resize(img, (width,height), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)


    for i in coor:
        x = i[0]
        y = i[1]
        d = float(b[(gridy * y) - 10][(gridx * x) - 10])
        g = float(b[(gridy * y) - 15][(gridx * x) - 15])
        avgb = (d+g)/2
        if avgb > 200:
            break

    index = coor.index(i)

    if index == 0:
        print("AR Tag is facing downwards")
    elif index == 1:
        print("AR Tag is facing left")
    elif index == 2:
        print("AR Tag is facing upwards")
    elif index == 3:
        print("AR Tag is facing right")

    #if the index is zero that is the white block is at left top corner
    if index == 0:
        x = 5
        y = 5
        for i in range(4):
            d = float(b[(gridy * y) - 10][(gridx * x) - 10])
            g = float(b[(gridy * y) - 15][(gridx * x) - 15])
            avgb = (d+g)/2
            if i == 0:
                x = x - 1
            elif i == 1:
                y = y - 1
            elif i == 2:
                x += 1

            if avgb > 200:
                reverse_code.append(1)
            else:
                reverse_code.append(0)

    #if the index is one that is the white block is at top right corner
    elif index == 1:
        x = 4
        y = 5
        for i in range(4):
            d = float(b[(gridy * y) - 10][(gridx * x) - 10])
            g = float(b[(gridy * y) - 15][(gridx * x) - 15])
            avgb = (d+g)/2
            if i == 0:
                y = y - 1
            elif i == 1:
                x += 1
            elif i == 2:
                y += 1

            if avgb > 200:
                reverse_code.append(1)
            else:
                reverse_code.append(0)

    #if the index is two that is the white block is at bottom right corner
    elif index == 2:
        x = 4
        y = 4
        for i in range(4):
            d = float(b[(gridy * y) - 10][(gridx * x) - 10])
            g = float(b[(gridy * y) - 15][(gridx * x) - 15])
            avgb = (d+g)/2
            if i == 0:
                x += 1
            elif i == 1:
                y += 1
            elif i == 2:
                x = x - 1

            if avgb > 200:
                reverse_code.append(1)
            else:
                reverse_code.append(0)

    #if the index is three that is the white block is at left down corner
    elif index == 3:
        x = 5
        y = 4
        for i in range(4):
            d = float(b[(gridy * y) - 10][(gridx * x) - 10])
            g = float(b[(gridy * y) - 15][(gridx * x) - 15])
            avgb = (d+g)/2
            if i == 0:
                y = y + 1
            elif i == 1:
                x = x - 1
            elif i == 2:
                y = y - 1

            if avgb > 200:
                reverse_code.append(1)
            else:
                reverse_code.append(0)

    return index

##
    # brief Finds the coordinates bounded by four corners
    # details This function finds all the coordinates inside the four corners
    #         of the square
    #
    # @param coordinates List of four coordinates of the detected square
    # @param canvas current frame from the video
    # return numpy array of all the internal points
##
def floodfill(coordinates,canvas):
    f_coor = coordinates.copy()
    f_coor = f_coor.astype(np.uint64)
    siz = canvas.shape

    canvas1 = np.zeros((siz[0],siz[1]))
    canvas2 = np.zeros((siz[0],siz[1]))

    cv2.line(canvas1,tuple(f_coor[0]), tuple(f_coor[1]),255,1)
    cv2.line(canvas1,tuple(f_coor[1]), tuple(f_coor[2]),255,1)
    cv2.line(canvas1,tuple(f_coor[2]), tuple(f_coor[3]),255,1)
    cv2.line(canvas1,tuple(f_coor[3]), tuple(f_coor[0]),255,1)
    miny = np.argmin(f_coor[:,1])
    maxy = np.argmax(f_coor[:,1])
    row_min = f_coor[miny,1]
    row_max = f_coor[maxy,1]
    row_min = row_min.astype(np.uint64)
    row_max = row_max.astype(np.uint64)
    all_rows = np.arange(row_min,row_max+1)
    all_rows = all_rows.astype(np.uint64)
    end_points = np.where(canvas1[all_rows]>0)
    end_points = np.array(end_points)
    shifted_end_points = np.insert(end_points,0,0,axis = 1)
    shifted_end_points = np.delete(shifted_end_points,-1,1)
    #print("end_points ",end_points)
    #print("shifted_end_points ",shifted_end_points)
    ##print("number of rows ",row_max-row_min)
    ##print(len(shifted_end_points[0]))
    diff = end_points - shifted_end_points
    #print(diff)
    #one_row = np.where(diff[0]>0)
    #print("one_row ",one_row)
    gap_loc = np.where(diff[1]>1)
    fill_coor_final = end_points.copy()
    ##print("pre final ", fill_coor_final)
    fill_coor_final[0] = fill_coor_final[0] + row_min
    ##print("final ",fill_coor_final)
    points_at_boundary = np.where(diff[1]==1)
    ##print("points at points_at_boundary ",points_at_boundary)
    ##print(gap_loc)
    ones_t = np.ones((1,len(points_at_boundary[0])))
    boundary_row = (end_points[0,points_at_boundary[0]] + row_min) * ones_t
    boundary_cols = end_points[1,points_at_boundary[0]]
    boundary_cols = np.reshape(boundary_cols, (1,boundary_cols.shape[0]))
    boundary = np.concatenate((boundary_cols,boundary_row), axis=0)
    ##print("boundary ",boundary)
    #gap_points = np.arange(end_points[1,gap_loc[0][1]-1],end_points[1,gap_loc[0][1]])
    #print("final ",gap_loc)
    internal_points = np.array([[0],[0]])
    for i in range(len(gap_loc[0])):
        if i == 0:
            continue
        internal = np.arange(end_points[1,gap_loc[0][i]-1],(end_points[1,gap_loc[0][i]])+1)
        internal = np.reshape(internal,(1,internal.shape[0]))
        #print(internal.shape)
        ones_u = np.ones((1,internal.shape[1]))
        #print(ones_u.shape)
        row = (end_points[0,gap_loc[0][i]]+row_min) * ones_u
        #print(row.shape)
        #fill_coor = np.concatenate((row, internal), axis = 0)
        #print(fill_coor)
        #row
        #print("fill shape ",fill_coor.shape)
        #fill_coor_final = np.insert(fill_coor_final,0,fill_coor,axis = 1)
        #remaining_rows = np.insert(fill_coor_final[0],0,row)
        #remaining_cols = np.insert(fill_coor_final[1],0,internal)
        #remaining_rows = np.reshape(remaining_rows,(1,remaining_rows.shape[0]))
        #remaining_cols = np.reshape(remaining_cols,(1,remaining_cols.shape[0]))
        fill_coor = np.concatenate((internal,row),axis = 0)
        internal_points = np.concatenate((internal_points,fill_coor),axis = 1)
        if i + 1 == len(gap_loc[0]):
            break
    #itr = row_max-row_min
    #for i in range(itr):

    #poi = np.where(canvas1[row,:]>0)
    #print("poi ",poi)
    #print(f_coor[miny,1])
    #print("poi len ",len(poi[0]))
    #cv2.imshow("poi",canvas1)
    internal_points = internal_points[:,1:]             #to take care of extra coordinate assigned while variable declaration
    internal_points = np.concatenate((internal_points,boundary),axis=1)
    one_f = np.ones((1,internal_points.shape[1]))
    internal_points = np.concatenate((internal_points,one_f),axis = 0)
    internal_points = internal_points.astype(np.uint64)
    canvas2[internal_points[1,:],internal_points[0,:]] = 255
    #cv2.imshow("fill ",canvas2)
    #print(internal_points.shape)
    return internal_points

##
    # brief arranges the coordinates of the square in anticlockwise manner
    # details The coordinates are arranged in anticlockwise manner in the
    #         array to obtain the homography
    #
    # @param approx1 numpy array of the corners of the square
    # @param  all_tags1 list of all arranged coordinates
    # return arranged np.array of arranged coordinates
    # return all_tags1 list of all arranged coordinates
##
def arrange_the_coordinates(approx1, all_tags1):
    final = approx1
    final = final.reshape((4,2))
    #print(final)
    arranged = np.zeros((4,2))

    sq_final = np.square(final)
    sum_final = np.sum(sq_final, axis = 1)
    dist_final = np.sqrt(sum_final)
    min_ind = np.argmin(dist_final)
    arranged[0] = final[min_ind]
    #print(dist_final)
    new_final = np.delete(final,min_ind,0)
    dist_final = np.delete(dist_final,min_ind,0)

    max_ind = np.argmax(dist_final)
    arranged[2] = new_final[max_ind]
    new_final = np.delete(new_final,max_ind,0)

    maxy = np.argmax(new_final[:,1])
    arranged[1] = new_final[maxy]
    new_final = np.delete(new_final,maxy,0)

    arranged[3] = new_final[0]

    return arranged, all_tags1

##
    # brief Checks if the coordinates are arranged properly
    # details Checks if the coordinates are arranged by checking the point
    #         of intersection of the diagonals. If the point does not match
    #         then the arrangement is changed.
    #
    # @param arrng partially arranged coordinates
    # return np.array of arranged coordinates
##
def check_if_the_coordinates_are_arranged_properly(arrng):
    score = 0
    centx1 = (arrng[0,0] + arrng[2,0])/2
    centy1 = (arrng[0,1] + arrng[2,1])/2
    centx2 = (arrng[1,0] + arrng[3,0])/2
    centy2 = (arrng[1,1] + arrng[3,1])/2
    if centx1-4 <= centx2 <= centx1+4 and centy1-4 <= centy2 <= centy1+4:
        score = score + 1
    if score == 0:
        arrng1 = arrng.copy()
        arrng[1] = arrng1[2]
        arrng[2] = arrng1[1]
    return arrng

##
    # brief Removes the inner most contour of the Artag boundary shape
    # details Removes a special case of error in arranging coordinates
    #         by checking the point of intersection of diagonals
    #
    # @param arrng np.array of partially arranged coordinates
    # return np.array and index corresponding to whether properly arranged
##
def final_check_of_coordinates(arrng):
    score = 0
    centx1 = (arrng[0,0] + arrng[2,0])/2
    centy1 = (arrng[0,1] + arrng[2,1])/2
    centx2 = (arrng[1,0] + arrng[3,0])/2
    centy2 = (arrng[1,1] + arrng[3,1])/2
    if centx1-4 <= centx2 <= centx1+4 and centy1-4 <= centy2 <= centy1+4:
        score = score + 1
    if score == 0:
        return arrng, 1
    return arrng, 0


##
    # brief Checks if the detected contour is inside the outside contour
    # details In the detected contours this function checks if the contour
    #         is inside the outside big contour of the black square.
    #
    # @param approx1 numpy array of the arranged corners of the square
    # @param  all_tags1 list of all arranged coordinates
    # return int corresponding to whether the detected contour is the required
    #        one.
##
def check_if_the_coordinates_are_inside_the_outside_square(approx1, all_tags1):
    for i in all_tags1:
        diff = i - approx1
        diff_sq = np.square(diff)
        dist_sq = np.sum(diff_sq, axis = 1)
        gh = np.where(dist_sq<5041)
        score = len(gh[0])
        if score >= 1:
            return 1
    return 0


#out = cv2.VideoWriter('non_inter_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))

##
    # brief Main function that detects and calls all other functions
    # details Detects the contours,  obtains the ARtag and imposes the test
    #         image.
##
def read_img_and_video_and_get_coordinates():
    img_test = cv2.imread("testudo.png")

    #vid = cv2.VideoCapture("multipleTags.mp4")
    vid = cv2.VideoCapture("Tag1.mp4")
    count = 0
    while(True):
        r, frame = vid.read()
        if frame is None:
            break
        img = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
        #img = frame
        img1 =img.copy()
        img2 =img.copy()
        img3 =img.copy()
        siz = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ft = fft.fft2(gray)
        ftshif = fft.fftshift(ft)
        rows, cols = gray.shape

        centy = rows//2
        centx = cols//2
        bound = 6

        # filter LPF
        for y in range(len(ftshif)):
            for x in range(len(ftshif[0])):
                if y > (centy - bound) and y < (centy + bound) and x > (centx - bound) and  x < (centx + bound):
                    ftshif[y][x] = 0


        invftshif = fft.ifftshift(ftshif)
        invft = fft.ifft2(invftshif)
        invftcv = np.uint8(np.abs(invft))

        cv2.imshow("ifft",invftcv)

        can = cv2.Canny(invftcv, 70,250)
        cv2.imshow("canny",can)
        #cv2.waitKey(0)
        contour = cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        #print("len ", len(contour))
        contour = sorted(contour, key = cv2.contourArea, reverse = True)[:10]

        # filtering the required contours coordinates
        all_tags = list()
        for i in contour:
            perimeter=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.09*perimeter,True)
            area = cv2.contourArea(approx)

            if area<2200and area>50 and len(approx) == 4:
                arranged1, all_tags = arrange_the_coordinates(approx, all_tags)
                arranged1 = check_if_the_coordinates_are_arranged_properly(arranged1)
                arranged1, chck0 = final_check_of_coordinates(arranged1)
                if chck0 == 1:
                    continue
                chck1 = check_if_the_coordinates_are_inside_the_outside_square(arranged1, all_tags)
                if chck1 == 1:
                    continue
                all_tags.append(arranged1)


        cv2.imshow("ifft",invftcv)
        img12 = img.copy()
        for i in range(len(all_tags)):
            flood_coor = floodfill(all_tags[i],img)
            orien = get_ARtag(all_tags[i],img)
            #img12 = impose_test_image(all_tags[i],img_test,orien, img12)
            img12 = impose_interpolated_test_image(all_tags[i],img_test,orien, img12, flood_coor)
        cv2.imshow("impose1",img12)
        cv2.waitKey(1)
        #out.write(img12)

    #out.release()
    vid.release()
    return

##
    # brief Performs bilinear interpolation
    # details Performs bilinear interpolation for non interger pixel values
    #
    # @param cor1 np.array of all the points inside the four corner points
    # @param  img14 contains the image.
    # return np.array of interpolated pixel values.
##
def bilinear_interpolation(cor, img14):
    sh = np.shape(cor)
    pix_val = np.zeros((sh[1],3))
    cor_xy = cor[:2,:]

    up_x = np.ceil(cor[0,:]).astype(np.uint64)
    up_y = np.ceil(cor[1,:]).astype(np.uint64)

    down_x = np.floor(cor[0,:]).astype(np.uint64)
    down_y = np.floor(cor[1,:]).astype(np.uint64)

    a = cor_xy[0,:] - down_x
    b = cor_xy[1,:] - down_y

    wt_top_right = (a*b).reshape((sh[1],1))
    wt_top_left = ((1-a)*b).reshape((sh[1],1))
    wt_down_left = ((1-a)*(1-b)).reshape((sh[1],1))
    wt_down_right = (a*(1-b)).reshape((sh[1],1))


    wt_top_right = np.concatenate((wt_top_right, wt_top_right, wt_top_right), axis = 1)
    wt_top_left = np.concatenate((wt_top_left, wt_top_left, wt_top_left), axis = 1)
    wt_down_left = np.concatenate((wt_down_left, wt_down_left, wt_down_left), axis = 1)
    wt_down_right = np.concatenate((wt_down_right, wt_down_right, wt_down_right), axis = 1)

    pix_val = (wt_top_right*img14[up_y[:],up_x[:]]) + (wt_top_left*img14[up_y[:],down_x[:]]) + (wt_down_left*img14[down_y[:],down_x[:]]) + (wt_down_right*img14[down_y[:],up_x[:]])

    pix_val[pix_val>255] = 255
    pix_val = pix_val.astype(np.uint8)
    return pix_val


##
    # brief Gets the ARtag
    # details Obtains the birds eye view of the ARtag
    #
    # @param arranged np.array of the arranged corners of the square
    # @param  img contains the current frame from the video
    # return int corresponding to the detected orientation of the ARtag
##
def get_ARtag(arranged,img):
    shp = img.shape
    artag_new = np.zeros((160,160,3))
    interpolated_artag = np.zeros((160,160,3))
    ho = artag_new.shape
    artag = [[0,0], [0,160], [160,160], [160,0]]
    A = np.zeros(shape=(8,9))
    i = 0

    for a in range(8):
        if a%2 == 0:
            A[a,:] = [artag[i][0], artag[i][1], 1, 0, 0, 0, -(arranged[i][0] * artag[i][0]), -(arranged[i][0] * artag[i][1]), -arranged[i][0]]
        else:
            A[a,:] = [0, 0, 0, artag[i][0], artag[i][1], 1, -(arranged[i][1] * artag[i][0]), -(arranged[i][1] * artag[i][1]), -arranged[i][1]]
            i += 1

    U,sigma,V = np.linalg.svd(A)

    Vt = V.T
    h = Vt[:,8]/Vt[8][8]

    f = 0
    H = np.zeros(shape=(3,3))
    for i in range(3):
        for j in range(3):
            H[i][j] = h[f]
            f += 1
    #Hinv = np.linalg.inv(H)
    new_img_coor = np.ones((ho[0], ho[1]))

    new_coor = list()

    for i in range(ho[0]):
        wid = np.arange(ho[1])
        wid = np.reshape(wid, (1,ho[0]))
        one_tp = np.ones((1,ho[0]))
        ro = i * one_tp
        coor = np.concatenate((wid,ro,one_tp), axis = 0)
        new_img_coor = np.dot(H,coor)
        new_img_coor = new_img_coor/new_img_coor[2]

        new_img_coor[0,new_img_coor[0,:]>shp[1]-1] = shp[1]-1
        new_img_coor[1,new_img_coor[1,:]>shp[0]-1] = shp[0]-1

        pixel_values = bilinear_interpolation(new_img_coor, img)
        interpolated_artag[i,:] = pixel_values
        new_img_coor = new_img_coor.astype(np.uint64)

        artag_new[i,:] = img[new_img_coor[1,:],new_img_coor[0,:]]
    artag_new=artag_new.astype(np.uint8)
    interpolated_artag=interpolated_artag.astype(np.uint8)
    #orien = orientation(artag_new)          # return the corresponding index of the orientation from the function
    # return the corresponding index of the orientation from the function
    orien = orientation(interpolated_artag)
    #cv2.imshow("impose",artag_new)
    cv2.imshow("interpolated_tag",interpolated_artag)
    cv2.imshow("video",img)
    return orien

##
    # brief Imposes the testudo image
    # details Imposes the testudo image on the main video frame
    #
    # @param arranged np.array of the arranged corners of the square
    # @param img_test contains the image of the testudo
    # @param orien contains the orientation index
    # @param img12 contians the main video frame
    # return np.array frame with imposed image
##
def impose_test_image(arranged, img_test, orien, img12):
    shp = img12.shape
    ho = img_test.shape

    if orien == 3:
        world = [[0,ho[0]], [ho[1],ho[0]], [ho[1],0], [0,0]]
    elif orien == 0:
        world = [[ho[1],ho[0]], [ho[1],0], [0,0], [0,ho[0]]]
    elif orien == 1:
        world = [[ho[1],0], [0,0], [0,ho[0]], [ho[1],ho[0]]]
    elif orien == 2:
        world = [[0,0], [0,ho[0]], [ho[1],ho[0]], [ho[1],0]]
    image = list()
    image = arranged

    A = np.zeros(shape=(8,9))
    i = 0
    for a in range(8):
        if a%2 == 0:
            A[a,:] = [world[i][0], world[i][1], 1, 0, 0, 0, -(image[i][0] * world[i][0]), -(image[i][0] * world[i][1]), -image[i][0]]
        else:
            A[a,:] = [0, 0, 0, world[i][0], world[i][1], 1, -(image[i][1] * world[i][0]), -(image[i][1] * world[i][1]), -image[i][1]]
            i += 1

    U,sigma,V = np.linalg.svd(A)
    Vt = V.T
    h = Vt[:,8]/Vt[8][8]            # normalizing the matrix

    f = 0
    H = np.zeros(shape=(3,3))
    for i in range(3):                  # reshaping the homography matrix
        for j in range(3):
            H[i][j] = h[f]
            f += 1

    new_img = np.zeros((ho[0], ho[1]))
    new_coor = list()

    for i in range(ho[0]):
        wid = np.arange(ho[1])
        wid = np.reshape(wid, (1,ho[1]))
        one_tp = np.ones((1,ho[1]))
        ro = i * one_tp
        coor = np.concatenate((wid,ro,one_tp), axis = 0)
        new_img_coor = np.dot(H,coor)
        new_img_coor = new_img_coor/new_img_coor[2]

        new_img_coor[0,new_img_coor[0,:]>shp[1]-1] = shp[1]-1
        new_img_coor[1,new_img_coor[1,:]>shp[0]-1] = shp[0]-1

        new_img_coor = new_img_coor.astype(np.uint64)

        img12[new_img_coor[1,:],new_img_coor[0,:]] = img_test[i,:]

    #cv2.imshow("impose1",img)
    #cv2.waitKey(1)

    return img12

##
    # brief Imposes the testudo image
    # details Imposes the testudo image on the main video frame
    #
    # @param arranged np.array of the arranged corners of the square
    # @param img_test contains the image of the testudo
    # @param orien contains the orientation index
    # @param img12 contians the main video frame
    # @param flood contains all the coordinates bounded by the four corners
    #        of the square
    # return np.array frame with interpolated imposed image
##
def impose_interpolated_test_image(arranged, img_test, orien, img12, flood):
    shp = img12.shape
    ho = img_test.shape

    if orien == 3:
        world = [[0,ho[0]], [ho[1],ho[0]], [ho[1],0], [0,0]]
    elif orien == 0:
        world = [[ho[1],ho[0]], [ho[1],0], [0,0], [0,ho[0]]]
    elif orien == 1:
        world = [[ho[1],0], [0,0], [0,ho[0]], [ho[1],ho[0]]]
    elif orien == 2:
        world = [[0,0], [0,ho[0]], [ho[1],ho[0]], [ho[1],0]]
    image = list()
    image = arranged

    ###### the world coordinates are arranged in the form of cartesian [x,y]
    #####  and care is later taken while assigning pixel values
    A = np.zeros(shape=(8,9))
    i = 0
    for a in range(8):
        if a%2 == 0:
            A[a,:] = [world[i][0], world[i][1], 1, 0, 0, 0, -(image[i][0] * world[i][0]), -(image[i][0] * world[i][1]), -image[i][0]]
        else:
            A[a,:] = [0, 0, 0, world[i][0], world[i][1], 1, -(image[i][1] * world[i][0]), -(image[i][1] * world[i][1]), -image[i][1]]
            i += 1


    U,sigma,V = np.linalg.svd(A)
    Vt = V.T
    h = Vt[:,8]/Vt[8][8]            # normalizing the matrix

    f = 0
    H = np.zeros(shape=(3,3))

    # reshaping the homography matrix
    for i in range(3):
        for j in range(3):
            H[i][j] = h[f]
            f += 1

    dup = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
    dup1 = dup.copy()

    points = np.array(image, dtype = np.int32)
    points = [points]

    cv2.fillPoly(dup, points, (0,0,0))
    artag1 = dup1 - dup
    tag_points = np.where(artag1>0)
    #print("fillPoly points ",len(tag_points[0]))
    #cv2.imshow("new1 ",img12)
    new_img = np.zeros((ho[0], ho[1]))
    new_coor = list()

    x_c = np.reshape(tag_points[1][:],(1,len(tag_points[0])))
    y_c = np.reshape(tag_points[0][:],(1,len(tag_points[0])))
    one_c = np.ones((1,len(tag_points[0])))
    final_points = np.concatenate((x_c, y_c, one_c), axis = 0)

    final_points = final_points.astype(np.uint64)

    H_inv = np.linalg.inv(H)
    #new_test_points = np.dot(H_inv,final_points)
    new_test_points = np.dot(H_inv,flood)
    new_test_points = new_test_points/new_test_points[2]

    new_test_points[0,new_test_points[0,:]>ho[1]-1] = ho[1]-1
    new_test_points[1,new_test_points[1,:]>ho[0]-1] = ho[0]-1

    pixel_values = bilinear_interpolation(new_test_points, img_test)
    #img12[final_points[1,:],final_points[0,:]] = pixel_values
    img12[flood[1,:],flood[0,:]] = pixel_values

    #cv2.imshow("new ",artag1)
    return img12

##
    # Initiates the code
##
read_img_and_video_and_get_coordinates()
cv2.destroyAllWindows()

from scipy import fft
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from ImposeImage import get_ARtag
from KalmanFilter import KalmanFilter
from KalmanFilter import KalmanFilterLocation


#used to compute the norm
def magnitude(mat):
    sum = 0
    for i in mat:
        sum = sum + (i)**2
    root = sum**(1/2)
    return root

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
    arranged = np.zeros((4,2))

    sq_final = np.square(final)
    sum_final = np.sum(sq_final, axis = 1)
    dist_final = np.sqrt(sum_final)
    min_ind = np.argmin(dist_final)
    arranged[0] = final[min_ind]

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

# out = cv2.VideoWriter('Cube1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920,1080))

def main():
    all_camera_locations = list()
    all_camera_rotations = list()

    vid = cv2.VideoCapture("..\\Tag1.mp4")
    kf = KalmanFilter(model_varianceX=20, model_varianceY=20, measurement_stdX=1.1, measurement_stdY=1.1, dt=0.1)
    kf_cam_loc = KalmanFilterLocation(model_varianceX=20, model_varianceY=20, model_varianceZ=20, measurement_stdX=0.5, measurement_stdY=0.5, measurement_stdZ=0.5, dt=0.1)
    
    count = 0
    while(True):
        r, frame = vid.read()
        if frame is None:
            break
        # img = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
        img = frame
        siz = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ft = fft.fft2(gray)
        ftshif = fft.fftshift(ft)
        rows, cols = gray.shape

        #dimensions of the filter to be applied
        centy = rows//2
        centx = cols//2
        bound = 5

        #filter HPF
        for y in range(len(ftshif)):
            for x in range(len(ftshif[0])):
                if y > (centy - bound) and y < (centy + bound) and x > (centx - bound) and  x < (centx + bound):
                    ftshif[y][x] = 0

        # magnitude_spectrum = np.log(np.abs(ftshif) + 1)
        # plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.show()

        invftshif = fft.ifftshift(ftshif)
        invft = fft.ifft2(invftshif)
        invftcv = np.uint8(np.abs(invft))

        # cv2.imshow("fft", invftcv)

        can = cv2.Canny(invftcv, 90,800)        #used canny edge detector and then passed the resulting image to findContours

        # cv2.imshow("canny", can)

        error = False
        contour=cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour=imutils.grab_contours(contour)
        contour = sorted(contour, key = cv2.contourArea, reverse = True)[:5]



        #filtering the required contours coordinates to get the corners of the ARtag
        detected = False
        all_tags = list()
        for i in contour:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.09*perimeter, True)
            area = cv2.contourArea(i)
            # print("area ",area)
            if area<17000 and area>2000 and len(approx) == 4:
                arranged, all_tags = arrange_the_coordinates(approx, all_tags)
                arranged = check_if_the_coordinates_are_arranged_properly(arranged)
                arranged, chck0 = final_check_of_coordinates(arranged)
                if chck0 == 1:
                    error = True
                    # continue
                chck1 = check_if_the_coordinates_are_inside_the_outside_square(arranged, all_tags)
                if chck1 == 1:
                    error = True
                pred, map = kf.prediction()

                # correction step only if we have a measurement
                if error == False:
                    est = kf.correction(arranged)
                # if count>10:
                #     exit()
                if error == True:
                    # continue
                    pred = np.dot(map, pred)
                    est = np.zeros((4,2))
                    est[:,0] = np.reshape(pred[:4], (1,4))
                    est[:,1] = np.reshape(pred[4:], (1,4))
                
                # uncomment to enable kalman filter tracking
                # arranged = est
                detected = True
                break

        
        if (detected == False):
            # out.write(img)
            cv2.imshow("cube ",img)
            cv2.waitKey(1)
            continue
        

        orientation = get_ARtag(arranged, img)

        # now the coordinats of the ARtag are saved in the list named arranged and starting with the problem to get the pose matrix
        K = np.array([[1406.08415449821, 0, 0],[2.20679787308599, 1417.99930662800, 0],[1014.13643417416, 566.347754321696, 1]])
        K = K.T

        cube_length = 100
        cube = [[0, 0, 0, 1], [0, cube_length, 0, 1], [cube_length, cube_length, 0, 1], [cube_length, 0, 0, 1], [0, 0, -cube_length, 1], [0, cube_length, -cube_length, 1], [cube_length, cube_length, -cube_length, 1],[cube_length, 0, -cube_length, 1]]
        if orientation == 3:
            world = [[0, cube_length], [cube_length, cube_length], [cube_length, 0], [0, 0]]
        elif orientation == 0:
            world = [[cube_length, cube_length], [cube_length, 0], [0, 0], [0, cube_length]]
        elif orientation == 1:
            world = [[cube_length, 0], [0, 0], [0, cube_length], [cube_length, cube_length]]
        elif orientation == 2:
            world = [[0, 0], [0, cube_length], [cube_length, cube_length], [cube_length, 0]]
        
        A = np.zeros(shape=(8,9))
        i = 0
        for a in range(8):
            if a%2 == 0:
                A[a,:] = [world[i][0], world[i][1], 1, 0, 0, 0, -(arranged[i][0] * world[i][0]), -(arranged[i][0] * world[i][1]), -arranged[i][0]]
            else:
                A[a,:] = [0, 0, 0, world[i][0], world[i][1], 1, -(arranged[i][1] * world[i][0]), -(arranged[i][1] * world[i][1]), -arranged[i][1]]
                i += 1

        U,sigma,Vt = np.linalg.svd(A)

        V = Vt.T
        h = V[:,8]/V[8][8]

        f = 0
        H = np.zeros(shape = (3,3))
        for i in range(3):                      # reshaping the homography matrix
            for j in range(3):
                H[i][j] = h[f]
                f += 1

        K_inv = np.linalg.inv(K)
        B_tilde = np.dot(K_inv, H)
        lambd = (2/(magnitude(B_tilde[:,0]) + magnitude(B_tilde[:,1])))

        B = lambd * B_tilde
        if np.linalg.det(B_tilde) < 0:
            B = (-1) * B
        r1 = B[:, 0].reshape((1,3))
        r2 = B[:, 1].reshape((1,3))
        r3 = np.cross(r1, r2)


        pred_loc, map_loc = kf_cam_loc.prediction()

        # doing a transpose since r1, r2, r3 are of shape [1,3] and are concatenated along axis 0
        R = np.concatenate((r1,r2,r3), axis = 0).T
        t = B[:, 2].reshape((3,1))

        # correcting translation using kalman tracker
        t = kf_cam_loc.correction(t)

        RT_mat = np.concatenate((R,t), axis = 1)
        pose = np.dot(K,RT_mat) 


        new_coor = list()
        for i in range(len(cube)):                  #multiplying the coordinates with the pose matrix
            new_c = np.dot(pose,np.array(cube[i]).T)
            new_x = new_c[0]/new_c[2]
            new_y = new_c[1]/new_c[2]
            new_coor.append((int(new_x),int(new_y)))

        #drawing the cube on the image with the new computed coordinates
        img = cv2.line(img, new_coor[0], new_coor[1], (0,0,255), 2)
        img = cv2.line(img, new_coor[1], new_coor[2], (0,0,255), 2)
        img = cv2.line(img, new_coor[2], new_coor[3], (0,0,255), 2)
        img = cv2.line(img, new_coor[3], new_coor[0], (0,0,255), 2)
        img = cv2.line(img, new_coor[0], new_coor[4], (0,0,255), 2)
        img = cv2.line(img, new_coor[1], new_coor[5], (0,0,255), 2)
        img = cv2.line(img, new_coor[2], new_coor[6], (0,0,255), 2)
        img = cv2.line(img, new_coor[3], new_coor[7], (0,0,255), 2)
        img = cv2.line(img, new_coor[4], new_coor[5], (0,0,255), 2)
        img = cv2.line(img, new_coor[5], new_coor[6], (0,0,255), 2)
        img = cv2.line(img, new_coor[6], new_coor[7], (0,0,255), 2)
        img = cv2.line(img, new_coor[7], new_coor[4], (255,0,0), 2)
        cv2.imshow("cube ",img)
        # out.write(img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        

        camera_orientation = R.T
        camera_location = -camera_orientation.dot(t)

        all_camera_locations.append(camera_location.T)
        all_camera_rotations.append(R)


        count += 1

    # saving camera rotations and locations for visualizations
    all_camera_locations = np.array(all_camera_locations)
    all_camera_locations = np.squeeze(all_camera_locations, axis=1)
    np.savetxt("camera_locations.txt", all_camera_locations)
    
    all_camera_rotations = np.array(all_camera_rotations)

    all_camera_rotations = np.reshape(all_camera_rotations, (all_camera_rotations.shape[0]*all_camera_rotations.shape[1], all_camera_rotations.shape[2]))
    np.savetxt("camera_rotations.txt", all_camera_rotations)

    vid.release()
    # out.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()

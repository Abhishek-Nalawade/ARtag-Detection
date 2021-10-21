from scipy import fft
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


#used to compute the norm
def magnitude(mat):
    sum = 0
    for i in mat:
        sum = sum + (i)**2
    root = sum**(1/2)
    return root


#out = cv2.VideoWriter('Cube1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (640,480))

vid = cv2.VideoCapture("Tag1.mp4")
count = 0
while(True):
    r, frame = vid.read()
    if frame is None:
        break
    img = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
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

    invftshif = fft.ifftshift(ftshif)
    invft = fft.ifft2(invftshif)
    invftcv = np.uint8(np.abs(invft))

    can = cv2.Canny(invftcv, 90,800)        #used canny edge detector and then passed the resulting image to findContours

    contour=cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour=imutils.grab_contours(contour)
    contour = sorted(contour, key = cv2.contourArea, reverse = True)[:5]

    #filtering the required contours coordinates to get the corners of the ARtag
    for i in contour:
        perimeter=cv2.arcLength(i,True)
        approx=cv2.approxPolyDP(i,0.09*perimeter,True)
        area = cv2.contourArea(i)
        #print("area ",area)
        if area<4000and area>100 and len(approx) == 4:
            final = approx
            final=final.reshape(4,2)
            arranged=np.zeros((4,2))

            sq_final = np.square(final)
            sum_final = np.sum(sq_final, axis = 1)
            dist_final = np.sqrt(sum_final)
            min_ind = np.argmin(dist_final)
            arranged[0] = final[min_ind]
            #print(dist_final)
            new_final = np.delete(final,min_ind,0)


            minx = np.argmin(new_final[:,0])
            arranged[1] = new_final[minx]
            new_final = np.delete(new_final,minx,0)


            maxy = np.argmax(new_final[:,1])
            arranged[2] = new_final[maxy]
            new_final = np.delete(new_final,maxy,0)

            arranged[3] = new_final[0]
            break

    # now the coordinats of the ARtag are saved in the list named arranged and starting with the problem to get the pose matrix
    K = np.array([[1406.08415449821, 0, 0],[2.20679787308599, 1417.99930662800, 0],[1014.13643417416, 566.347754321696, 1]])
    K = K.T

    cube = [[0, 0, 0,1], [0, 100, 0,1], [100, 100, 0,1], [100, 0, 0,1], [0, 0, 100,1], [0, 100, 100,1], [100, 100, 100,1],[100, 0, 100,1]]
    A = np.zeros(shape=(8,9))
    i = 0
    for a in range(8):
        if a%2 == 0:
            A[a,:] = [cube[i][0], cube[i][1], 1, 0, 0, 0, -(arranged[i][0] * cube[i][0]), -(arranged[i][0] * cube[i][1]), -arranged[i][0]]
        else:
            A[a,:] = [0, 0, 0, cube[i][0], cube[i][1], 1, -(arranged[i][1] * cube[i][0]), -(arranged[i][1] * cube[i][1]), -arranged[i][1]]
            i += 1

    U,sigma,V = np.linalg.svd(A)

    Vt = V.T
    h = Vt[:,8]/Vt[8][8]

    f = 0
    H = np.zeros(shape = (3,3))
    for i in range(3):                      # reshaping the homography matrix
        for j in range(3):
            H[i][j] = h[f]
            f += 1


    B = np.dot(np.linalg.inv(K), H)
    if np.linalg.det(B) > 0:
        B = (-1) * B
    r1 = B[:, 0]
    r2 = B[:, 1]
    r1 = np.reshape(r1, (3,1))
    r2 = np.reshape(r2, (3,1))
    r3 = np.cross(B[:,0], B[:,1])
    t = B[:, 2]
    t = np.reshape(t, (3,1))
    lambd = (2/(magnitude(np.dot(np.linalg.inv(K),B[:,0])) + magnitude(np.dot(np.linalg.inv(K),B[:,1]))))
    #lambd = 1/magnitude(np.dot(np.linalg.inv(K),B[:,0]))
    r1 = lambd * r1
    #lambd = 1/magnitude(np.dot(np.linalg.inv(K),B[:,1]))
    r2 = lambd * r2
    r1_cross = np.reshape(r1,(1,3))
    r2_corss = np.reshape(r2, (1,3))
    r3 = np.cross(r1_cross,r2_corss)
    r3 = np.reshape(r3, (3,1))
    #lambd = 1/magnitude(np.dot(np.linalg.inv(K),B[:,2]))
    t = lambd * t
    RT_mat = np.concatenate((r1,r2,r3,t), axis = 1)
    pose = np.dot(K,RT_mat)

    new_coor = list()
    for i in range(len(cube)):                  #multiplying the coordinates with the pose matrix
        new_c = np.dot(pose,np.array(cube[i]).T)
        new_x = new_c[0]/new_c[2]
        new_y = new_c[1]/new_c[2]
        new_coor.append((int(new_x),int(new_y)))
    #print(new_coor)



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
    img = cv2.line(img, new_coor[7], new_coor[4], (0,0,255), 2)
    cv2.imshow("cube ",img)
    #out.write(img)
    cv2.waitKey(5)
    #cv2.waitKey(0)



vid.release()
cv2.destroyAllWindows()

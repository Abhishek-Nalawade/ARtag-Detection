import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, model_varianceX, model_varianceY, measurement_stdX, measurement_stdY, dt=0.1):
        # initial state
        self.mu_t_minus_1 = np.zeros((16, 1))
        # self.mu_t_minus_1[:8, :] = np.array([[322],[324],[349],[346],[191],[218],[219],[192]])
        self.mu_t_minus_1[:8, :] = np.array([[10],[10],[30],[30],[20],[50],[50],[20]])
        # need to rearrange input as [[x1],
        #                             [x2],
        #                             [x3],
        #                             [x4],
        #                             [y1],
        #                             [y2]...
        #                             [x1_dot],
        #                             [x2_dot]....
        #                             [y4_dot]]

        # self.model = np.array([[1, 0, 0, 0, 0, 0, 0, 0, dt, 0......],
        #                         [0, 1, 0, 0, 0, 0, 0, 0, 0, dt......],
        #                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0......],
        #                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0......],
        #                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0......],
        #                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0......],
        #                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0......],
        #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0......]])

        # model
        # x = x + x_dot*dt and y = y + y_dot*dt
        self.A = np.eye(16)        # 4 xcoor, 4 ycoor, 4 x_dot and 4 y_dot
        for i in range(int(self.A.shape[0]/2)):
            self.A[i, i+8] = -dt
        # print(self.A)

        self.C = np.zeros((8, self.A.shape[1]))
        temp = np.eye(8)
        self.C[:8, :8] = temp

        # process noise
        self.R = np.zeros((self.A.shape[0], self.A.shape[1]))
        variance_x_dot = model_varianceX
        variance_y_dot = model_varianceY
        # model noise
        # let Var(v) = E(v**2) - mu_v**2 = sigma_v**2
        # Var(x) = E(x**2) - mu_x**2 = E(v**2 * dt**2) - mu_v**2 * dt**2
        #      = dt**2 * (E(v**2) - mu_v**2) = dt**2 * sigma_v**2
        # COV(x,v) = E(xv) - mu_x * mu_v = E(v*dt * v) - mu_v*dt * mu_v
        #          = dt * (E(v**2) - mu_v**2) = dt * sigma_v**2
        for i in range(self.A.shape[0]):
            if i < self.A.shape[0]/2:
                self.R[i, i] = (dt**2) * variance_x_dot
                self.R[i, i+8] = dt * variance_x_dot
            else:
                self.R[i, i] = variance_x_dot
                self.R[i, i-8] = dt * variance_x_dot

        # measurement noise
        self.Q = np.zeros((8, 8))
        std_x = measurement_stdX
        std_y = measurement_stdY
        self.Q[:4, :4] = std_x**2
        self.Q[4:, 4:] = std_y**2

        self.sigma_t_minus_1 = np.eye(self.A.shape[0])

        self.mu_t_predicted = 1
        self.sigma_t_predicted = 1

    def prediction(self):
        # prediction
        self.mu_t_predicted = np.dot(self.A, self.mu_t_minus_1)
        self.sigma_t_predicted = np.dot(np.dot(self.A, self.sigma_t_minus_1), self.A.T) + self.R

        # # updating t-1 values if in case there is no measurement
        # self.mu_t_minus_1 = self.mu_t_predicted
        # self.sigma_t_minus_1 = self.sigma_t_predicted
        return self.mu_t_predicted, self.C

    def correction(self, z_t):
        arranged_z_t = np.zeros((z_t.shape[0]*z_t.shape[1], 1))
        arranged_z_t[:4] = np.reshape(z_t[:, 0], (4,1))
        arranged_z_t[4:] = np.reshape(z_t[:, 1], (4,1))
        # Kalman gain
        S = np.linalg.inv(np.dot(self.C, np.dot(self.sigma_t_predicted, self.C.T)) + self.Q)
        K = np.dot(np.dot(self.sigma_t_predicted, self.C.T), S)

        # correction
        self.mu_t = self.mu_t_predicted + np.dot(K, (arranged_z_t - np.dot(self.C, self.mu_t_predicted)))
        I = np.eye(self.C.shape[1])
        self.sigma_t = np.dot((I - np.dot(K, self.C)), self.sigma_t_predicted)

        # updating
        self.mu_t_minus_1 = self.mu_t
        self.sigma_t_minus_1 = self.sigma_t

        est = np.dot(self.C, self.mu_t)
        estimate = np.zeros((4,2))
        estimate[:,0] = np.reshape(est[:4], (1,4))
        estimate[:,1] = np.reshape(est[4:], (1,4))
        return estimate

class ExtendedKalmanFilter:
    def __init__(self, model_varianceX, model_varianceY, measurement_stdX, measurement_stdY, dt=0.1):
        # initial state
        self.mu_t_minus_1 = np.zeros((24, 1))
        # self.mu_t_minus_1[:8, :] = np.array([[322],[324],[349],[346],[191],[218],[219],[192]])
        self.mu_t_minus_1[:8, :] = np.array([[10],[10],[30],[30],[20],[50],[50],[20]])

        # model
        # x = x + x_dot*dt + 0.5*x_doubleDot*(dt**2)
        # x_dot = x_dot + x_doubleDot*dt
        self.A = np.eye(24)        # 4 xcoor, 4 ycoor, 4 x_dot and 4 y_dot, 4 x_doubleDot, 4 y_doubleDot
        for i in range(int((2/3) * self.A.shape[0])):
            if i < self.A.shape[0]/3:
                self.A[i, i+8] = dt
                self.A[i, i+16] = -(1/2) * (dt**2)
            elif self.A.shape[0]/3 <= i:
                self.A[i, i+8] = -dt

        # jacobian in this case is same as the model matrix
        self.G = self.A

        # process noise
        self.R = np.zeros((self.A.shape[0], self.A.shape[1]))
        variance_x_doubleDot = model_varianceX
        variance_y_doubleDot = model_varianceY

        for i in range(self.A.shape[0]):
            # xcoor and ycoor varinace with their respective _dot and _doubleDot
            if i < self.A.shape[0]/3:
                if i < self.A.shape[0]/6:
                    self.R[i, i] = (1/4) * (dt**4) * variance_x_doubleDot
                    self.R[i, i+8] = (1/2) * (dt**2) * variance_x_doubleDot
                    self.R[i, i+16] = (1/2) * dt * variance_x_doubleDot
                else:
                    self.R[i, i] = (1/4) * (dt**4) * variance_y_doubleDot
                    self.R[i, i+8] = (1/2) * (dt**2) * variance_y_doubleDot
                    self.R[i, i+16] = (1/2) * dt * variance_y_doubleDot
            elif self.A.shape[0]/3 <= i < (2/3)*self.A.shape[0]:     # x_dot and y_dot variance with their respective coor and _doubleDot
                if i < self.A.shape[0]/2:
                    self.R[i,i] = (dt**2) * variance_x_doubleDot
                    self.R[i, i+8] = dt * variance_x_doubleDot
                    self.R[i, i-8] = (1/2) * (dt**2) * variance_x_doubleDot
                else:
                    self.R[i,i] = (dt**2) * variance_y_doubleDot
                    self.R[i, i+8] = dt * variance_y_doubleDot
                    self.R[i, i-8] = (1/2) * (dt**2) * variance_y_doubleDot
            else:           # x_doubleDot and y_doubleDot variance with their respective coor and _dot
                if i < (5/6)*self.A.shape[0]:
                    self.R[i, i] = variance_x_doubleDot
                    self.R[i, i-8] = dt * variance_x_doubleDot
                    self.R[i, i-16] = (1/2) * dt * variance_x_doubleDot
                else:
                    self.R[i, i] = variance_y_doubleDot
                    self.R[i, i-8] = dt * variance_y_doubleDot
                    self.R[i, i-16] = (1/2) * dt * variance_y_doubleDot

        self.H = np.zeros((8, self.A.shape[1]))
        temp = np.eye(8)
        self.H[:8, :8] = temp

        # measurement noise
        self.Q = np.zeros((8, 8))
        std_x = measurement_stdX
        std_y = measurement_stdY
        self.Q[:4, :4] = std_x**2
        self.Q[4:, 4:] = std_y**2
        # print(self.Q)

        self.sigma_t_minus_1 = np.eye(self.A.shape[0])

        self.mu_t_predicted = 1
        self.sigma_t_predicted = 1

    def prediction(self):
        self.mu_t_predicted = np.dot(self.A, self.mu_t_minus_1)
        self.sigma_t_predicted = np.dot(np.dot(self.G, self.sigma_t_minus_1), self.G.T) + self.R

        # updating t-1 values if in case there is no measurement
        self.mu_t_minus_1 = self.mu_t_predicted
        self.sigma_t_minus_1 = self.sigma_t_predicted
        return self.mu_t_predicted, self.H

    def correction(self, z_t):
        arranged_z_t = np.zeros((z_t.shape[0]*z_t.shape[1], 1))
        arranged_z_t[:4] = np.reshape(z_t[:, 0], (4,1))
        arranged_z_t[4:] = np.reshape(z_t[:, 1], (4,1))
        # Kalman gain
        S = np.linalg.inv(np.dot(self.H, np.dot(self.sigma_t_predicted, self.H.T)) + self.Q)
        K = np.dot(np.dot(self.sigma_t_predicted, self.H.T), S)

        # correction
        self.mu_t = self.mu_t_predicted + np.dot(K, (arranged_z_t - np.dot(self.H, self.mu_t_predicted)))
        I = np.eye(self.H.shape[1])
        self.sigma_t = np.dot((I - np.dot(K, self.H)), self.sigma_t_predicted)

        # updating
        self.mu_t_minus_1 = self.mu_t
        self.sigma_t_minus_1 = self.sigma_t

        est = np.dot(self.H, self.mu_t)
        # print(est.shape)
        # print(self.mu_t.shape)
        estimate = np.zeros((4,2))
        estimate[:,0] = np.reshape(est[:4], (1,4))
        estimate[:,1] = np.reshape(est[4:], (1,4))
        return estimate

class KalmanFilterLocation:
    def __init__(self, model_varianceX, model_varianceY, model_varianceZ, measurement_stdX, measurement_stdY, measurement_stdZ, dt=0.1):
        # initial state
        self.mu_t_minus_1 = np.zeros((6, 1))

        # need to rearrange input as [[cx],
        #                             [cy],
        #                             [cz],
        #                             [cx_dot],
        #                             [cy_dot],
        #                             [cz_dot]]

        # self.model = np.array([[1, 0, 0, dt, 0, 0],
        #                         [0, 1, 0, 0, dt, 0],
        #                         [0, 0, 1, 0, 0, dt],
        #                         [0, 0, 0, 1, 0, 0],
        #                         [0, 0, 0, 0, 1, 0],
        #                         [0, 0, 0, 0, 0, 1]])

        # model
        # cx = cx + cx_dot*dt, cy = cy + cy_dot*dt and cz = cz + cz_dot*dt
        self.A = np.eye(6)        # 3 location, 3 loc_dots
        for i in range(int(self.A.shape[0]/2)):
            self.A[i, i+3] = dt
        # print(self.A)

        # model covariance
        # let Var(v) = E(v**2) - mu_v**2 = sigma_v**2
        # Var(x) = E(x**2) - mu_x**2 = E(v**2 * dt**2) - mu_v**2 * dt**2
        #      = dt**2 * (E(v**2) - mu_v**2) = dt**2 * sigma_v**2
        # COV(x,v) = E(xv) - mu_x * mu_v = E(v*dt * v) - mu_v*dt * mu_v
        #          = dt * (E(v**2) - mu_v**2) = dt * sigma_v**2
        self.R = np.eye(6)
        variance_x_dot = model_varianceX
        variance_y_dot = model_varianceY
        variance_z_dot = model_varianceZ

        self.R[0, 0] = variance_x_dot * (dt**2)
        self.R[0, 3] = variance_x_dot * dt
        self.R[3, 0] = variance_x_dot * dt
        self.R[3, 3] = variance_x_dot
        
        self.R[1, 1] = variance_y_dot * (dt**2)
        self.R[1, 4] = variance_y_dot * dt
        self.R[4, 1] = variance_y_dot * dt
        self.R[4, 4] = variance_y_dot

        self.R[2, 2] = variance_z_dot * (dt**2)
        self.R[2, 5] = variance_z_dot * dt
        self.R[5, 2] = variance_z_dot * dt
        self.R[5, 5] = variance_z_dot
        # print(self.R)
        
        # observation model
        self.C = np.zeros((3, self.A.shape[1]))
        self.C[:self.C.shape[0], :self.C.shape[0]] = np.eye(self.C.shape[0])
        
        # measurement covariance
        self.Q = np.zeros((3, 3))
        self.Q[0, 0] = measurement_stdX**2
        self.Q[1, 1] = measurement_stdX**2
        self.Q[2, 2] = measurement_stdX**2

        self.sigma_t_minus_1 = np.eye(self.A.shape[0])

        self.mu_t_predicted = 1
        self.sigma_t_predicted = 1

    def prediction(self):
        self.mu_t_predicted = np.dot(self.A, self.mu_t_minus_1)
        self.sigma_t_predicted = np.dot(self.A, np.dot(self.sigma_t_minus_1, self.A.T)) + self.R
        return self.mu_t_predicted, self.C

    def correction(self, z_t):
        # # updating
        # self.mu_t_minus_1 = self.mu_t
        # self.sigma_t_minus_1 = self.sigma_t

        # est = np.dot(self.C, self.mu_t)
        # estimate = np.zeros((4,2))
        # estimate[:,0] = np.reshape(est[:4], (1,4))
        # estimate[:,1] = np.reshape(est[4:], (1,4))

        # Kalman gain
        S = np.linalg.inv(np.dot(self.C, np.dot(self.sigma_t_predicted, self.C.T)) + self.Q)
        K = np.dot(self.sigma_t_predicted, np.dot(self.C.T, S))

        # correction
        mu_t = self.mu_t_predicted + np.dot(K, (z_t - np.dot(self.C, self.mu_t_predicted)))
        I = np.eye(self.C.shape[1])
        sigma_t = np.dot((I - np.dot(K, self.C)), self.sigma_t_predicted)

        # updating for next iteration
        self.mu_t_minus_1 = mu_t
        self.sigma_t_minus_1 = sigma_t
        return  np.dot(self.C, mu_t)

import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, model_varianceX, model_varianceY, measurement_stdX, measurement_stdY, dt=0.1):
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


        self.A = np.eye(16)        # 4 xcoor, 4 ycoor, 4 x_dot and 4 y_dot
        for i in range(int(self.A.shape[0]/2)):
            self.A[i, i+8] = dt
        # print(self.A)

        self.C = np.zeros((8, self.A.shape[1]))
        temp = np.eye(8)
        self.C[:8, :8] = temp

        # process noise
        self.R = np.zeros((self.A.shape[0], self.A.shape[1]))
        variance_x_dot = model_varianceX
        variance_y_dot = model_varianceY
        for i in range(self.A.shape[0]):
            if i < self.A.shape[0]/2:
                self.R[i, i] = (dt**2) * variance_x_dot
                self.R[i, i+8] = dt * variance_x_dot
            else:
                self.R[i, i] = variance_x_dot
                self.R[i, i-8] = dt * variance_x_dot

        # measurement noise
        self.Q = np.ones((8, 8))
        std_x = measurement_stdX
        std_y = measurement_stdY
        self.Q[:4, :4] = self.Q[:4, :4] * std_x**2
        self.Q[4:, 4:] = self.Q[4:, 4:] * std_y**2

        self.sigma_t_minus_1 = np.eye(self.A.shape[0])

        self.mu_t_predicted = 1
        self.sigma_t_predicted = 1

    def prediction(self):
        # prediction
        self.mu_t_predicted = np.dot(self.A, self.mu_t_minus_1)
        self.sigma_t_predicted = np.dot(np.dot(self.A, self.sigma_t_minus_1), self.A.T) + self.R
        return self.mu_t_predicted, self.C

    def correction(self, z_t, img):
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

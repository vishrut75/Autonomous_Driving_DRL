import airsim
import numpy as np
import cv2
import time


class Car_Environment():
    def __init__(self):
        #self.image_shape = image_shape
        #self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient()

    def _setup_car(self):
        self.car.reset()
        self.car.confirmConnection()
        self.car.enableApiControl(True)

        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def __del__(self):
        self.car.reset()

    def _do_action(self,action):
        linear_vel = 2*action[0] - 1;
        steer = action[1]-0.5;
        print(linear_vel,steer)
        if linear_vel >= 0:
            self.car_controls.throttle = linear_vel
            self.car_controls.brake = 0
        else:
            self.car_controls.throttle = 0
            self.car_controls.brake = -1*linear_vel
        self.car_controls.steering = steer
        self.car.setCarControls(self.car_controls)
        time.sleep(0.1)

    def pause(self):
        self.car.simPause(True)
        print("Simulation Paused")

    def resume(self):
        self.car.simPause(False)
        print("Simulation Resumed")

    def observe_lidar(self):
        lidarData = self.car.getLidarData();
        points = np.array(lidarData.point_cloud,dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        print("\tReading: time_stamp: %d number_of_points: %d" % (lidarData.time_stamp, len(points)))
        print("\t\tlidar position: %s" % (lidarData.pose.position))
        print("\t\tlidar orientation: %s" % (lidarData.pose.orientation))

        return points

    def observe_img(self):
        self.image_request = self.car.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene,False,False)])
        response = self.image_request[0]

        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        self.depth_request = self.car.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis,False,False)])
        response = self.depth_request[0]

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        dpt_rgb = img1d.reshape(response.height, response.width, 3)
        dpt_gray = cv2.cvtColor(dpt_rgb,cv2.COLOR_BGR2GRAY)
        
        shape = np.shape(dpt_gray)
        print(shape)
        dpt_gray = np.reshape(dpt_gray,[shape[0],shape[1],1])
        
        vis = np.concatenate((img_rgb, dpt_gray), axis=2)
        
        return vis

    def _compute_reward(self):
        MAX_SPEED = 200
        MIN_SPEED = 1


        reward = (self.car_state.speed - MIN_SPEED)*(MAX_SPEED - self.car_state.speed)/400

        done = 0
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
                reward = -100
        if self.state["collision"]:
            done = 1
            reward = -150

        return reward, done



    def get_obs(self):
        image = self.observe_img()

        self.car_state = self.car.getCarState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided
        return image, [self.state["pose"].linear_velocity.get_length(),self.state["pose"].angular_velocity.get_length()]

    def step(self, action):
        self._do_action(action)
        obs, vel = self.get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, vel


    def reset(self):
        self._setup_car()
        self._do_action([1,0])
        return self.get_obs()

# Car = Car_Environment()
# Car.reset()
# Car.observe_lidar()


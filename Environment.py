import airsim
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


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
        _ = self.car.simSetSegmentationObjectID("Road[\w]*", 42, True);

    def _setup_car(self):
        self.car.reset()
        self.car.confirmConnection()
        self.car.enableApiControl(True)

        self.car_controls = airsim.CarControls()
        self.car_state = self.car.getCarState()
        self.car_controls.brake = 0
        self.car_controls.throttle = 0
        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def __del__(self):
        self.car.reset()

    def _do_action(self,action):
        linear_vel = float(action[0])
        # steer = float((0.5*action[1])/(1+0.2*self.car_state.speed))
        steer = float(action[1]-0.5)
        # print(linear_vel,steer)
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
        # print("Simulation Paused")

    def resume(self):
        self.car.simPause(False)
        # print("Simulation Resumed")

    def observe_lidar(self):
        
        lidarData = self.car.getLidarData();
        points = np.array(lidarData.point_cloud,dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        # print("\tReading: time_stamp: %d number_of_points: %d" % (lidarData.time_stamp, len(points)))
        # print("\t\tlidar position: %s" % (lidarData.pose.position))
        # print("\t\tlidar orientation: %s" % (lidarData.pose.orientation))
        lid_array = [60]*25
        for p in points:
            ang = np.arctan2(p[0],p[2])*180/np.pi
            lid_array[round(ang)//5 - 6] = np.linalg.norm(p)
        return lid_array

    def observe_movement(self):
        for i in range(4):
            self.depth_request = self.car.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis,False,False)])
            response = self.depth_request[0]

            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            dpt_rgb = img1d.reshape(response.height, response.width, 3)
            dpt_gray = cv2.cvtColor(dpt_rgb,cv2.COLOR_BGR2GRAY)
            shape = np.shape(dpt_gray)
            dpt_gray = np.reshape(dpt_gray,[shape[0],shape[1],1])
            
            if i==0:
                vis = dpt_gray
            else:
                 vis = np.concatenate((vis, dpt_gray), axis=2)
            time.sleep(0.1)
        return vis

    
    def observe_img(self,visual=False,depth=False):
        self.image_request = self.car.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene,False,False)])
        response = self.image_request[0]

        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        img_rgb = cv2.resize(img_rgb,dsize=(128,72), interpolation=cv2.INTER_CUBIC)

        if depth or visual:
            self.depth_request = self.car.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis,False,False)])
            response = self.depth_request[0]

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            dpt_rgb = img1d.reshape(response.height, response.width, 3)
            dpt_gray = cv2.cvtColor(dpt_rgb,cv2.COLOR_BGR2GRAY)
        
            dpt_gray = cv2.resize(dpt_gray,dsize=(128, 72), interpolation=cv2.INTER_CUBIC)

            shape = np.shape(dpt_gray)
            dpt_gray = np.reshape(dpt_gray,[shape[0],shape[1],1])
        
            # vis = np.concatenate((img_rgb, dpt_gray), axis=2)

        if(visual):
            cv2.imshow("img",img_rgb)
            cv2.waitKey(0)
            cv2.imshow("img",dpt_gray)
            cv2.waitKey(0)
        
        return img_rgb


    def segment(self,visual=False):
        img_rgb = np.array([])
        while img_rgb.size==0:
            self.image_request = self.car.simGetImages([airsim.ImageRequest(1, airsim.ImageType.Segmentation,False,False)])
            response = self.image_request[0]

            # get numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(response.height, response.width, 3)

        img_rgb = cv2.resize(img_rgb,dsize=(128,72), interpolation=cv2.INTER_CUBIC)

        road_color = np.array([106,31,92])
        mask = cv2.inRange(img_rgb, road_color,road_color)
        if(visual):
            cv2.imshow("img",img_rgb)
            cv2.waitKey(1)

        count = 0
        for i in mask:
            for j in i:
                if(j==255):
                    count+=1

        self.car.simPrintLogMessage("Road_count: ",str(count))
        return count, img_rgb

    def _compute_reward(self):
        MAX_SPEED = 9.0
        MIN_SPEED = 1.0
        road_count, _ = self.segment()
        road_reward = np.arctan((road_count - 3500)/200)*40/np.pi

        reward = float((self.car_state.speed - MIN_SPEED)*(MAX_SPEED - self.car_state.speed)*6.0/9.0) + road_reward

        done = 0
        stale = 0
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 0.1:
                done = 0
                stale = 1
                reward = -6.0
        if self.state["collision"]:
            done = 1
            reward = -16.0*self.car_state.speed


        return reward, done, stale



    def get_obs(self):
        _, image = self.segment()
        state = self.observe_lidar()
        self.car_state = self.car.getCarState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided
        state.append(self.state["pose"].linear_velocity.get_length())
        state.append(self.state["pose"].angular_velocity.get_length())
        return image, state

    def step(self, action):
        self._do_action(action)
        image, state = self.get_obs()
        reward, done, stale = self._compute_reward()

        return image, reward, done, state, stale


    def reset(self):
        self._setup_car()
        self._do_action([0,0.5])
        return self.get_obs()


# Car = Car_Environment()
# Car.reset()
# Car.observe_img(True)
#while True:
#    Car.observe_lidar()
#    Car._do_action([0.5,np.random.random()])
#    time.sleep(1)
#    print(Car.segment(True))


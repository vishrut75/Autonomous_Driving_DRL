import airsim
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import math


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
        linear_vel = float(action[0]+1)/2.0
        #steer = float(action[1]-0.5)
        steer = float(action[1])/2.0
        # print(linear_vel,steer)
        #if linear_vel >= 0:
        self.car_controls.throttle = linear_vel
        self.car_controls.brake = 0
        #else:
        #    self.car_controls.throttle = 0
        #    self.car_controls.brake = -1*linear_vel
        #if linear_vel<0.1:
        #    self.car_controls.brake = 1.0
        self.car_controls.steering = steer
        self.car.setCarControls(self.car_controls)
        #time.sleep(0.1)

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
        lid_array = [1]*25
        for p in points:
            ang = np.arctan2(p[0],p[2])*180/np.pi
            lid_array[round(ang)//5 - 6] = (np.linalg.norm(p))/30
        return lid_array

    def observe_movement(self):
        for i in range(1):
            self.depth_request = self.car.simGetImages([airsim.ImageRequest("Cam0", airsim.ImageType.Segmentation,False,False)])
            response = self.depth_request[0]

            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            dpt_rgb = img1d.reshape(response.height, response.width, 3)
            road_color = np.array([106,31,92])
            mask = cv2.inRange(dpt_rgb, road_color,road_color)
            dpt_gray = cv2.resize(mask,dsize=(32,18), interpolation=cv2.INTER_CUBIC)
            #dpt_gray = cv2.cvtColor(dpt_rgb,cv2.COLOR_BGR2GRAY)
            shape = np.shape(dpt_gray)
            dpt_gray = np.reshape(dpt_gray,[shape[0],shape[1],1])
            if i==0:
                vis = dpt_gray
            else:
                 vis = np.concatenate((vis, dpt_gray), axis=2)
            #time.sleep(0.01)
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

    def get_image(self,name,gray=False):
        img_rgb = np.array([])
        while img_rgb.size==0:
            self.image_request = self.car.simGetImages([airsim.ImageRequest(name, airsim.ImageType.Segmentation,False,False)])
            response = self.image_request[0]

            # get numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(response.height, response.width, 3)

        img_rgb = cv2.resize(img_rgb,dsize=(128,72), interpolation=cv2.INTER_CUBIC)

        if gray:
            dpt_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
            return dpt_gray
        
        return img_rgb


    def segment(self,visual=False):
        
        img_rgb = self.get_image("Cam0") 
        road_color = np.array([106,31,92])
        mask = cv2.inRange(img_rgb, road_color,road_color)
        #dpt_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
        d1 = self.get_image("Cam1",False)
        d2 = self.get_image("Cam2",False)
        dpt_gray = np.concatenate((d1,img_rgb),axis=1)
        dpt_gray = np.concatenate((dpt_gray,d2),axis=1)
        dpt_gray = cv2.inRange(dpt_gray, road_color,road_color)
        if(visual):
            cv2.imshow("img",dpt_gray)
            cv2.waitKey(1)

        count = 0
        for i in mask:
            for j in i:
                if(j==255):
                    count+=1

        x_s,y_s = np.shape(dpt_gray)
        dpt_gray = np.reshape(dpt_gray,[x_s,y_s,1])
        self.car.simPrintLogMessage("Road_count: ",str(count))
        return count, dpt_gray

    def sign(self,val):
        if(val>=0):
            return 1
        return -1

    def _compute_dist(self):
        BETA = 4
        THRESH_DIST = 2
        pnts = [
            np.array([x, y, z, w])
            for x, y, z, w in [
                (-122, -1,122 ,-1), (128, -128, 128, 128), (128, 128, -128, 128), (128, -128, -128, -128),(0, -128, 0, 128), (-128, -128, -128, 128)
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()
        dist = 10000000
        ang = 0
        vec = 0
        i = 0
        for idx in range(0, len(pnts)):
            j = pnts[idx]
            pts = np.array([[j[0],j[1],0],[j[2],j[3],0]])
            dist_now = np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1])))/np.linalg.norm(pts[i] - pts[i + 1])
            if pts[i][0]==pts[i+1][0] and car_pt[1] <= max(pts[i+1][1],pts[i][1]) and car_pt[1] >= min(pts[i+1][1],pts[i][1]):
                dist_now = abs(car_pt[0]-pts[i][0])
            elif pts[i][1]==pts[i+1][1] and car_pt[0] <= max(pts[i+1][0],pts[i][0]) and car_pt[0] >= min(pts[i+1][0],pts[i][0]):
                dist_now = abs(car_pt[1]-pts[i][1])
            dist = min(dist, dist_now)
            if(dist==dist_now):
                ang = math.atan2(pts[i+1][1]-pts[i][1],pts[i+1][0]-pts[i][0])
                ang = ang*180/math.pi
                vec = self.sign(np.cross((car_pt - pts[i]),(pts[i+1]-pts[i]))[2])*dist

        if dist > THRESH_DIST:
            reward_dist = -1
        else:
            reward_dist =  -1*dist**2/BETA#math.cos(dist)#2*math.exp(-BETA*dist) - 1

        yaw = self.quat_to_yaw(self.state["pose"].orientation)
        delta_ang = (ang - yaw)%360
        if(delta_ang>180):
            delta_ang = delta_ang - 360
        delta_ang = delta_ang/180
        vec = round(vec*self.sign(math.cos(delta_ang*math.pi)),4)
        self.car.simPrintLogMessage("dist: ",str(vec))
        self.car.simPrintLogMessage("x: ",str(car_pt[0]))
        self.car.simPrintLogMessage("y: ",str(car_pt[1]))
        return reward_dist, vec, delta_ang

    def _compute_reward(self):
        MAX_SPEED = 7.0
        MIN_SPEED = 1.0
        road_count, _ = self.segment()
        # road_reward = np.arctan((road_count - 3500)/200)*40/np.pi
        #road_reward = ((road_count - 3500)/500)*2/3
        road_reward, _, _ = self._compute_dist()

        speed_reward = float((self.car_state.speed - MIN_SPEED)*(MAX_SPEED - self.car_state.speed)*1.0/9.0)
        #if self.car_state.speed>5:
        #    speed_reward = 0.5
        #if self.car_state.speed>7:
        #    speed_reward = 0.5

        if(road_reward<=-1):
            reward = -1
        else:
           reward= speed_reward + road_reward

        done = 0
        stale = 0
        if self.car_state.speed <= 0.1:
            done = 0
            stale = 1
            reward = -1.0
            
        #if reward>1.0:
        #    reward = 1.0
        if reward<-1.0:
            reward = -1.0
        
        if road_count < 2500:
            done = 1
            reward = -2.0

        if self.state["collision"]:
            done = 1
            reward = -2.0
            
        return reward, done, stale
        #if reward <=-1.0:
        #    return np.array([reward/2.0, reward/2.0]), done, stale

        #return np.array([speed_reward/2.0,road_reward/2.0]), done, stale

    def quat_to_yaw(self,quat):
        qy = quat.y_val
        qx = quat.x_val
        qz = quat.z_val
        qw = quat.w_val
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw*180/math.pi

    def get_obs(self):
        #_, image = self.segment()
        image = self.observe_movement()
        state = self.observe_lidar()
        self.car_state = self.car.getCarState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided
        #state.append(self.state["pose"].linear_velocity.get_length())
        #state.append(self.state["pose"].angular_velocity.get_length())
        _, dist, delta_ang = self._compute_dist()
        state.append(self.car_state.speed/6)
        state.append(delta_ang)
        state.append(dist/3)
        #print(delta_ang)
        #print(self.state["pose"])
        return image, state

    def step(self, action):
        self._do_action(action)
        image, state = self.get_obs()
        reward, done, stale = self._compute_reward()

        return image, reward, done, state, stale


    def reset(self):
        self._setup_car()
        self._do_action([1.0,0.5])
        time.sleep(0.5)
        return self.get_obs()

#print("Started")
#Car = Car_Environment()
#image_request = Car.car.simGetImages([airsim.ImageRequest("Cam0", airsim.ImageType.Segmentation,False,False)])
#response = image_request[0]

## get numpy array
#img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

## reshape array to 4 channel image array H X W X 4
#img_rgb = img1d.reshape(response.height, response.width, 3)
#road_color = np.array([106,31,92])
#mask = cv2.inRange(img_rgb, road_color,road_color)
#cv2.imshow("img",mask)
#cv2.waitKey(0)
#Car._setup_car()
#Car.reset()
#Car.get_obs()
#Car.step([1,0])
#Car.observe_movement()
### Car.observe_img(True)
#while True:
###    Car.observe_lidar()
###    Car._do_action([0.5,np.random.random()])
###    time.sleep(1)
#   Car.segment(True)
    #Car.get_obs()


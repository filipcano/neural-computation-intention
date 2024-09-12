import gymnasium as gym
import numpy as np
import pandas as pd
import pygame, json
import time


def clipped_increase(value, increase, lowboud, upbound):
     value = value + increase
     value = min(value, upbound)
     value = max(value, lowboud)
     return value


class Car:
    def __init__(self, x=0, y=0, v=0, pygame_scale=10):
        self.x = x
        self.y = y
        self.v = v
        self.pygame_scale = pygame_scale

    def get_rect_object(self):
        center_x = self.x*self.pygame_scale
        center_y = self.y*self.pygame_scale
        buff_x = int(self.pygame_scale)
        buff_y = int(self.pygame_scale/2)
        # rect: left, top, width, height
        obj = pygame.Rect(center_x - buff_x, center_y + buff_y, 2*buff_x, 2*buff_y)
        return obj



class Ped:
    def __init__(self, x=0, y=0, pygame_scale=10):
        self.x = x
        self.y = y
        self.pygame_scale = pygame_scale
    
    def get_rect_object(self):
        center_x = self.x*self.pygame_scale
        center_y = self.y*self.pygame_scale
        buff_x = int(self.pygame_scale/3)
        buff_y = int(self.pygame_scale/3)
        # rect: left, top, width, height
        obj = pygame.Rect(center_x - buff_x, center_y + buff_y, 2*buff_x, 2*buff_y)
        return obj



            

class TrafficEnvironment(gym.Env):
    def __init__(self, paramsfile):
            super(TrafficEnvironment, self).__init__()
            with open(paramsfile, 'r') as fp:
                  params = json.load(fp)
            self.min_street_length = params["min_street_length"]
            self.max_street_length = params["max_street_length"]
            self.sidewalk_height = params["sidewalk_height"]
            self.crosswalk_pos = params["crosswalk_pos"]
            self.crosswalk_width = params["crosswalk_width"]
            self.max_speed_car = params["max_speed_car"]
            self.max_speed_ped = params["max_speed_ped"]
            self.max_accel_car = params["max_accel_car"]
            self.slippery_range_start = params["slippery_range_start"]
            self.slippery_range_end = params["slippery_range_end"]
            self.slippery_factor = params["slippery_factor"]
            self.car_x_init = params["car_x_init"]
            self.car_v_init = params["car_v_init"]
            self.ped_x_init = params["ped_x_init"]
            self.ped_y_init = params["ped_y_init"]
            self.avg_ped_vel = params["avg_ped_vel"]
            self.std_ped_vel = params["std_ped_vel"]
            self.car_width = params["car_width"]
            self.car_height = params["car_height"]
            self.car_y = params["car_y"]

            self.pedestrian_crossed = False

            self.do_render = True
            self.pygame_scale = 25

            self.max_street_y = self.sidewalk_height + self.crosswalk_width

            self.action_space =  gym.spaces.Box(low = np.array([-1, -1, -1]), high = np.array([1,1,1]), dtype=np.int32)
            obs_space_lowbounds = np.array([self.min_street_length, 0, self.min_street_length, 0])# pos_car, vel_car, pos_x_ped, pos_y_ped
            obs_space_upbounds = np.array([self.max_street_length, self.max_speed_car, self.max_street_length, self.max_street_y]) # pos_car, vel_car, pos_x_ped, pos_y_ped

            # print(obs_space_lowbounds)
            # print(obs_space_upbounds)

            self.observation_space = gym.spaces.Box(low = obs_space_lowbounds, high = obs_space_upbounds, dtype=np.int32) # Note: Box includes low and high, ie., closed interevals

            self.car = Car(x=self.car_x_init, y = self.car_y, v = self.car_v_init, pygame_scale=self.pygame_scale)
            self.ped = Ped(x = self.ped_x_init, y = self.ped_y_init, pygame_scale= self.pygame_scale)


            acc_probs = np.zeros(self.max_accel_car+1)
            acc_probs[0] = 0
            acc_probs[1] = 0.5
            for i in range(2,len(acc_probs)):
                acc_probs[i] = 0.5**(i-1)
            acc_probs = acc_probs/np.sum(acc_probs)

            acc_probs_slippery = acc_probs.copy()
            for i in range(1,len(acc_probs_slippery)):
                acc_probs_slippery[i] = acc_probs[i]/(np.power(2, i+self.slippery_factor)-1)
            acc_probs_slippery[0] = 1-np.sum(acc_probs_slippery[1:])

            self.acc_probs = acc_probs
            self.acc_probs_slippery = acc_probs_slippery

            print(acc_probs, acc_probs_slippery)

            if self.render:
                pygame.font.init()
            self.screen = None
            self.myfont = None
            self.clock = None

    
    
    def is_slippery(self):
         return (self.car.x >= self.slippery_range_start) and (self.car.x <= self.slippery_range_end)
    
    def gen_observation(self):
        obs = np.array([self.car.x, self.car.v, self.ped.x, self.ped.y])
        return obs

    def values_from_state(self, obs):
        car_x = obs[0]
        car_v = obs[1]
        ped_x = obs[2]
        ped_y = obs[3]
        return car_x, car_v, ped_x, ped_y

    def is_collision(self, prev_obs, next_obs):
        prev_car_x, prev_car_v, prev_ped_x, prev_ped_y = self.values_from_state(prev_obs)
        next_car_x, next_car_v, next_ped_x, next_ped_y = self.values_from_state(next_obs)

        ped_car_aligned = prev_ped_y == self.car.y or next_ped_y == self.car.y
        ped_before_car= prev_car_x <= max(prev_ped_x, next_ped_x)
        ped_after_car = next_car_x >= min(prev_ped_x, next_ped_x)

        return ped_car_aligned and ped_before_car and ped_after_car


    def car_reward(self, prev_obs, next_obs, car_action):
        #TODO: maybe add car_action to the reward, to reward noops (efficiency)
        if self.is_collision(prev_obs, next_obs):
            return -1
        next_car_x = next_obs[0]
        if next_car_x == self.max_street_length:
            return 1
        return -0.001
    

    
    def ped_reward(self, prev_obs, next_obs):
        if self.is_collision(prev_obs, next_obs):
            return -1
        next_car_x, next_car_v, next_ped_x, next_ped_y = self.values_from_state(next_obs)
        ped_on_init_sidewalk = next_ped_y <= self.sidewalk_height
        ped_on_final_sidewalk = next_ped_y >= self.sidewalk_height + self.crosswalk_width
        ped_on_crosswalk = (not (ped_on_init_sidewalk or ped_on_final_sidewalk)) and  (self.crosswalk_pos <= next_ped_x) and (next_ped_x <= self.crosswalk_pos+self.crosswalk_width)

        if ped_on_final_sidewalk: 
            if self.pedestrian_crossed:
                return 0
            self.pedestrian_crossed = True
            return 10
        if ped_on_init_sidewalk:
            return -0.1
        if ped_on_crosswalk:
            return -0.01
        # the only other possibility here is pedestrian on the road, outside sidewalk
        return -0.5


    def step(self, action):
        car_action = action[0]
        ped_action_x = action[1]
        ped_action_y = action[2] 
        prev_obs = self.gen_observation()
        if car_action == 0: # noop
            accel = 0 if np.random.rand() > 0.5 else -1
        else:
            acc_probs = self.acc_probs_slippery if self.is_slippery() else self.acc_probs
            accel_absolute = np.random.choice(np.arange(self.max_accel_car+1), p=acc_probs)
            accel = accel_absolute if car_action == 1 else -accel_absolute
        self.car.v = clipped_increase(self.car.v, accel, 0, self.max_speed_car)
        self.car.x = clipped_increase(self.car.x, self.car.v, self.min_street_length, self.max_street_length)


        self.ped.x = clipped_increase(self.ped.x, ped_action_x, self.min_street_length, self.max_street_length)
        self.ped.y = clipped_increase(self.ped.y, ped_action_y, 0 ,self.max_street_y)

        next_obs = self.gen_observation()

        finish_line = self.car.x == self.max_street_length
        collision = self.is_collision(prev_obs, next_obs)
        reward = (self.car_reward(prev_obs, next_obs, car_action), self.ped_reward(prev_obs, next_obs))
        done = finish_line or collision
        info = {"finish_line": finish_line, "collision" : collision}
        return next_obs, reward, done, info
    
    def reset(self):
        if self.do_render:
            self.screen = pygame.display.set_mode([int((self.max_street_length)*self.pygame_scale*1.1), int(self.max_street_y*self.pygame_scale*1.1)])
            self.myfont = pygame.font.SysFont('Helvetica', 14)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Traffic simulation")

        car_x_init = self.min_street_length
        car_v_init = np.random.randint(0, self.max_speed_car)
        self.car = Car(x=car_x_init, y = self.car_y, v = car_v_init, pygame_scale=self.pygame_scale)

        ped_x_init = np.random.randint(self.crosswalk_pos-self.crosswalk_width, self.crosswalk_pos+self.crosswalk_width)
        self.ped = Ped(x = ped_x_init, y = self.ped_y_init, pygame_scale= self.pygame_scale)
        self.pedestrian_crossed = False
        info = {}
        return self.gen_observation(), info
    
    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 0, 0), self.car.get_rect_object())
        pygame.draw.rect(self.screen, (0, 255, 0), self.ped.get_rect_object())
        pygame.draw.line(self.screen, (255,255,0),
                         start_pos=(self.max_street_length*self.pygame_scale, 0),
                         end_pos=(self.max_street_length*self.pygame_scale, self.max_street_y*self.pygame_scale))
        pygame.draw.line(self.screen, (255, 255, 0), 
                         start_pos=(0, self.sidewalk_height*self.pygame_scale),
                         end_pos=(self.max_street_length*self.pygame_scale, self.sidewalk_height*self.pygame_scale))
        pygame.draw.line(self.screen, (255, 255, 0), 
                         start_pos=(0, (self.sidewalk_height+self.crosswalk_width)*self.pygame_scale),
                         end_pos=(self.max_street_length*self.pygame_scale, (self.sidewalk_height+self.crosswalk_width)*self.pygame_scale))
        pygame.draw.rect(self.screen, (0,0,255),
                         pygame.Rect(
                             self.slippery_range_start*self.pygame_scale,
                             (self.sidewalk_height+0.5)*self.pygame_scale,
                             (self.slippery_range_end-self.slippery_range_start)*self.pygame_scale,
                             (self.crosswalk_width-2)*self.pygame_scale
                         ), 
                         width=1)
        # rect: left, top, width, height
        
        for i in range(self.sidewalk_height,9):
            height = 1.5*i-0.5
            pygame.draw.rect(self.screen, (255,255,255), 
                             pygame.Rect(self.crosswalk_pos*self.pygame_scale,
                                     height*self.pygame_scale,
                                     self.crosswalk_width*self.pygame_scale,
                                     self.pygame_scale),
                            width=1)

        pygame.display.update()
        time.sleep(0.05)


        

        

        



          


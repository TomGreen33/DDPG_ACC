import numpy as np
import copy
import pandas as pd
DT = 0.1

class Env(object):
    def __init__(self, TTC_threshold):
        self.action_bound = 2
        self.num_actions = 1
        
        self.penalty = 1 # penalty for collisions
        self.num_states = 3
        self.TTC_threshold = TTC_threshold
        self.action_upper_bound = 2.6
        self.action_lower_bound = -2.6
        self.last_action = 0

    def reset(self, episode):
        self.total_distance = episode["Leading Vehicle Speed"].sum()/10
        self.time_step = 0 # starting form 0 to n
        self.leader_speed = episode["Leading Vehicle Speed"] # Is part of the environment, i.e., not controlled by the actor, not part of sim
        self.episode_length = len(episode)
        self.sim_b2b_distance = np.zeros(len(episode))
        self.sim_follower_speed = np.zeros(len(episode))
        self.sim_b2b_distance[0] = episode["Bumper to Bumper Distance"].iloc[0] # initialize with initial spacing
        self.sim_follower_speed[0] = episode["Following Vehicle Speed"].iloc[0] # initialize with initial speed

        # State history stored as dataframe so that new states may be concatenated
        self.state_history = episode.head(1)
        # Current state stored as dictionary where values are floats
        self.current_state = episode[["Bumper to Bumper Distance", "Following Vehicle Speed", "Relative Speed"]].iloc[0].to_dict()
        
        self.is_collision = 0
        self.is_stall = 0
        self.time_len = len(episode)


        relative_speed = self.current_state["Relative Speed"]
        if relative_speed <= 0:
            relative_speed = 0.00001
            
        b2b_distance = self.current_state["Bumper to Bumper Distance"]
        
        self.TTC = - b2b_distance / relative_speed
        return self.current_state

    def step(self, action):
        # update state
        
        
        
        follower_speed = self.current_state["Following Vehicle Speed"] + action * DT
        
        self.time_step += 1
        leader_speed = self.leader_speed.iloc[self.time_step]
        
        if follower_speed <= 0:
            follower_speed = 0.00001
            self.is_stall = 1
        else:
            self.is_stall = 0

        relative_speed = leader_speed - follower_speed
        
        # Update b2b distance with relative speed times timestep 
        b2b_distance = self.current_state["Bumper to Bumper Distance"] + relative_speed * DT
        
        self.update_current_state(b2b_distance, follower_speed, relative_speed)
        
        current_state_df = self.current_state.copy()
        
        #judge collision and back
        if b2b_distance < 0:
            self.is_collision = 1

        #store the space history for error calculating
        self.sim_b2b_distance[self.time_step-1] = b2b_distance
        self.sim_follower_speed[self.time_step-1] = follower_speed

        # caculate the reward

        
        # f_gap_keeping = -(e_gap_keeping / self.leader_speed.iloc[:self.time_step].sum()/10) + 1
        # f_gap_keeping = -((e_gap_keeping-self.total_distance) / self.total_distance) - 0.5 #/self.episode_length  

        
        
        jerk = (action - self.last_action) / DT
        
        follower_headway = b2b_distance / follower_speed
        # self.TTC = -b2b_distance / relative_speed  # negative sign because of relative speed sign
        
        # f_jerk = -(jerk ** 2)/3600   # the maximum range is change from -3 to 3 in 0.1 s, then the jerk = 60

        # f_acc = - action**2/60

        self.last_action = action

        # if self.TTC >= 0 and self.TTC <= self.TTC_threshold:
            # f_ttc = np.log(self.TTC/self.TTC_threshold) 
        # else:
            # f_ttc = 0

        # mu = 0.422618  
        # sigma = 0.43659
        # if follower_headway <= 0:
        #     f_headway = -1
        # else:
        #     f_headway = (np.exp(-(np.log(follower_headway) - mu) ** 2 / 
        #                         (2 * sigma ** 2)) / 
        #                  (follower_headway * sigma * np.sqrt(2 * np.pi)))

        desired_headway = 1 + follower_speed * 3 # could replace with leader speed
        
        e_p =  min(b2b_distance - desired_headway, 2)
        
        e_v = min(relative_speed, 1.5)
        E_P_MAX = 15
        E_V_MAX = 10
        REWARD_THRESHOLD = -0.4483
        A = 0.1
        B = 0.1
        C = 0.2
        lam = 5e-3
        # calculate the reward
        r_abs = -(abs(e_p/E_P_MAX) + A * abs(e_v / E_V_MAX) + B * abs(action) + C * abs(jerk / (2*self.action_upper_bound/0.1)))
        r_qua = -lam*((e_p**2) + A * (e_v ** 2) + B * (action ** 2) + C * ((jerk * 0.1) ** 2))
        if r_abs < REWARD_THRESHOLD:
            reward = r_abs
        else:
            reward = r_qua
        # reward = (f_gap_keeping - self.penalty * self.is_collision)
        
        
        current_state_df["Leading Vehicle Speed"] = leader_speed
        # current_state_df["desired_distance"] = desired_distance
        # current_state_df["e_gap_keeping"] = b2b_distance-desired_distance
        # current_state_df["e_gap_keeping_non_linear"] = e_gap_keeping
        current_state_df["penalty"] = -self.penalty * self.is_collision
        current_state_df["reward"] = reward
        current_state_df = pd.DataFrame(current_state_df, index = [self.time_step])
        self.state_history = pd.concat([self.state_history, current_state_df])

        # record reward info

        # judge the end
        if self.time_step == self.time_len - 1 or self.is_collision == 1:
            done = True
        else:
            done = False
            
        next_state = self.current_state

        return next_state, reward, done

    def update_current_state(self, b2b_distance, follower_speed, relative_speed):
        self.current_state["Bumper to Bumper Distance"] = b2b_distance
        self.current_state["Following Vehicle Speed"] = follower_speed
        self.current_state["Relative Speed"] = relative_speed

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:33:09 2023

@author: TOM3O
"""

import pygame
import aggregate_episodes
import pandas as pd
import numpy as np
import aggregate_episodes


episodes = aggregate_episodes.aggregate_episodes()
white = (255, 255, 255)
black = (0, 0, 0)
#%%
def animate(episode, episode_simulated, rewards, episode_num):

    pygame.init()
    img = pygame.image.load('car.png')
    img_2 = pygame.image.load('car_good.png')
    # img.convert()
    
    rect1 = img.get_rect()
    rect2 = img.get_rect()
    
    rect1_sim = img.get_rect()
    rect2_sim = img.get_rect()
    
    rect_desired = img_2.get_rect()
    
    IMAGE_WIDTH = 1000
    IMAGE_HEIGHT = 500
    
    dis = pygame.display.set_mode((1000, 500))
    
    font = pygame.font.Font('freesansbold.ttf', 32)
    human_text = font.render('Human Control', True, black)
    simulation_text = font.render('DDPG ACC', True, black)
    
    
    
    
    
    VEHICLE_HEIGHT_METERS = 2.5 #  # In metersIn meters
    VEHICLE_WIDTH_METERS = 5 #  # In metersIn meters
    
    total_distance = max(episode["Leading Vehicle Speed"].sum()/10 + 2*VEHICLE_WIDTH_METERS + episode["Bumper to Bumper Distance"][0], 
                         episode_simulated["Leading Vehicle Speed"].sum()/10 + 2*VEHICLE_WIDTH_METERS + episode_simulated["Bumper to Bumper Distance"][0]) 
                       # Division by 10 because of 10 Hz
    
    scale_factor = IMAGE_WIDTH / total_distance # pixels/meter
    
    rect1.x = 0
    rect2.x = (VEHICLE_WIDTH_METERS + episode["Bumper to Bumper Distance"][0])*scale_factor
    
    rect1_sim.x = 0
    rect2_sim.x = (VEHICLE_WIDTH_METERS + episode_simulated["Bumper to Bumper Distance"][0])*scale_factor
    rect_desired.x = 0
    
    img = pygame.transform.scale(img, (VEHICLE_WIDTH_METERS * scale_factor, VEHICLE_HEIGHT_METERS * scale_factor))
    img_2 = pygame.transform.scale(img_2, (VEHICLE_WIDTH_METERS * scale_factor, VEHICLE_HEIGHT_METERS * scale_factor))

    rect1.w = VEHICLE_WIDTH_METERS * scale_factor
    rect1.h = VEHICLE_HEIGHT_METERS * scale_factor
    rect2.w = VEHICLE_WIDTH_METERS * scale_factor
    rect2.h = VEHICLE_HEIGHT_METERS * scale_factor
    rect1_sim.w = VEHICLE_WIDTH_METERS * scale_factor
    rect1_sim.h = VEHICLE_HEIGHT_METERS * scale_factor
    rect2_sim.w = VEHICLE_WIDTH_METERS * scale_factor
    rect2_sim.h = VEHICLE_HEIGHT_METERS * scale_factor
    rect_desired.w = VEHICLE_WIDTH_METERS * scale_factor
    rect_desired.h = VEHICLE_HEIGHT_METERS * scale_factor
    
    VEHICLE_HEIGHT = VEHICLE_HEIGHT_METERS * scale_factor
    VEHICLE_WIDTH = VEHICLE_WIDTH_METERS * scale_factor
    rect1.y = 2*(IMAGE_HEIGHT / 3) - (VEHICLE_HEIGHT/2)
    rect2.y = 2*(IMAGE_HEIGHT / 3) - (VEHICLE_HEIGHT/2)
    
    rect1_sim.y = (IMAGE_HEIGHT / 3) - (VEHICLE_HEIGHT/2)
    rect2_sim.y = (IMAGE_HEIGHT / 3) - (VEHICLE_HEIGHT/2)
    rect_desired.y = (IMAGE_HEIGHT / 3) - (VEHICLE_HEIGHT/2)
    clock = pygame.time.Clock()
    direction = 'Right'
    
    
    def animate(rect, image):
        
        dis.blit(image, rect)
    
    
    for i in range(max([len(episode), len(episode_simulated)])):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        # keys = pygame.key.get_pressed()
        


        dis.fill((255, 255, 255))
        dis.blit(simulation_text, simulation_text.get_rect(center = (IMAGE_WIDTH // 2, 100)))
        dis.blit(human_text, human_text.get_rect(center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)))
        try:
            info_text = font.render("Episode: " + str(episode_num) + ", reward: " + str(rewards[i]), True, black)
        except:
            1==1
        dis.blit(info_text, info_text.get_rect(center = (IMAGE_WIDTH // 2, 50)))
        animate(rect1, img)
        animate(rect2, img)
        animate(rect_desired, img_2)
        animate(rect1_sim, img)
        animate(rect2_sim, img)
        pygame.display.flip()
        
        try:
            rect2.x += (episode["Leading Vehicle Speed"].iloc[i]/10) * scale_factor
            rect1.x = rect2.x - (episode["Bumper to Bumper Distance"].iloc[i] + VEHICLE_WIDTH_METERS) * scale_factor
        except:
            1==1
        try:
            rect2_sim.x += (episode_simulated["Leading Vehicle Speed"].iloc[i]/10) * scale_factor
            rect1_sim.x = rect2_sim.x - (episode_simulated["Bumper to Bumper Distance"].iloc[i] + VEHICLE_WIDTH_METERS) * scale_factor
        except:
            1==1
        try:
            rect_desired.x = max(0, rect2_sim.x - (episode_simulated["desired_distance"].iloc[i]) * scale_factor)
        except:
            1==1
        
        clock.tick(50)
    
    
"""Script for the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=5,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--trajectory',         default='hover',    type=str,           help='Specifies the trajectory for the PID controlled experiment (default: hover) (options: hover, forward, turns, circle, ascent)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3 # Circle flying radius
    
    traj = ARGS.trajectory
    if traj == None or traj == 'hover':    
        INIT_XYZS = np.array([[0, 0, 0]])
        INIT_RPY = np.array([[0, 0, 0]])
    if traj == 'forward' or traj == 'turns':
        INIT_XYZS = np.array([[0, 0, 0.5]])
        INIT_RPY = np.array([[0, 0, 0]])
    if traj == 'circle' or traj == 'ascent':
        INIT_XYZS = np.array([[R*np.cos((0/6)*2*np.pi+np.pi/2), R*np.sin((0/6)*2*np.pi+np.pi/2)-R, H+0*H_STEP]])
        INIT_RPY = np.array([[0, 0, 0]])
    if traj == 'ascent_rpy':
        INIT_XYZS = np.array([[R*np.cos((0/6)*2*np.pi+np.pi/2), R*np.sin((0/6)*2*np.pi+np.pi/2)-R, H+0*H_STEP]])
        INIT_RPY = np.array([[0, 0, 0]])
    if traj == 'loop' or traj == 'forwardloop':
        INIT_XYZS = np.array([[0, 0, 1]])
        INIT_RPY = np.array([[0, 0, 0]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPY,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize a circular trajectory ######################

    if traj == 'hover':  
        ## Without waypoints:
        NUM_WP = 1
        TARGET_POS = np.zeros((NUM_WP,3))  
        TARGET_POS[:] = INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2] + 1
        wp_counter = 0
        
        # With waypoints:
        # PERIOD = 1
        # NUM_WP = ARGS.control_freq_hz*PERIOD
        # TARGET_POS = np.zeros((NUM_WP,3))
        # for i in range(NUM_WP):
        #     TARGET_POS[i, :] = INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2] + i/NUM_WP
        # wp_counter = 0

    if traj == 'forward': 
        ## Without waypoints:
        # NUM_WP = 1
        # TARGET_POS = np.zeros((NUM_WP,3))  
        # TARGET_POS[:] = INIT_XYZS[0, 0] + 1, INIT_XYZS[0, 1], INIT_XYZS[0, 2]
        # wp_counter = 0
        
        # With waypoints:
        PERIOD = 1
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))
        for i in range(NUM_WP):
            TARGET_POS[i, :] = INIT_XYZS[0, 0] + i/NUM_WP, INIT_XYZS[0, 1], INIT_XYZS[0, 2]
        wp_counter = 0

    #TODO
    if traj == 'turns':
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))
        for i in range(NUM_WP):
            TARGET_POS[i, :] = (i/NUM_WP)*(2*np.pi) + INIT_XYZS[0, 0], np.sin((i/NUM_WP)*(2*np.pi)) + INIT_XYZS[0, 1], INIT_XYZS[0, 2]
        wp_counter = 0

    if traj == 'circle': #Play for 15 sec with --duration_sec 15
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))
        for i in range(NUM_WP):
            TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], INIT_XYZS[0, 2]
        wp_counter = 0

    if traj == 'ascent': #Play for 15 sec with --duration_sec 15
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))
        for i in range(NUM_WP):
            TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], i/NUM_WP + INIT_XYZS[0, 2]
        wp_counter = 0

    #TODO: Nose flying
    if traj == 'ascent_rpy': #Play for 15 sec with --duration_sec 15
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))
        TARGET_RPY = np.zeros((NUM_WP,3))
        TARGET_RPY[0, :] = INIT_RPY[0, :]
        for i in range(NUM_WP):
            xi = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0]
            yi = R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1]
            TARGET_POS[i, :] = xi, yi, i/NUM_WP + INIT_XYZS[0, 2]
        #     if i < NUM_WP-1:    
        #         TARGET_RPY[i+1, :] = 0, 0, (TARGET_RPY[i, 2] - 2*np.pi/NUM_WP if i < NUM_WP/2 else -TARGET_RPY[i, 2] + i*2*np.pi/NUM_WP)
            
            # TODO: Target rpy is somehow not taken into account by the drone
            # Handle bottom, right, top, left cases of circle
            if i == 0:
                TARGET_RPY[i, :] = 0, 0, 0
            if i == NUM_WP/4 -1:
                TARGET_RPY[i, :] = 0, 0, -np.pi/2
            if i == NUM_WP*2/4 -1:
                TARGET_RPY[i, :] = 0, 0, -np.pi
            if i == NUM_WP*3/4 -1:
                TARGET_RPY[i, :] = 0, 0, -np.pi*3/2
            if i == NUM_WP -1:
                TARGET_RPY[i, :] = 0, 0, 0
            # Standard case
            # Calculate zero point of tangent of point (x,y) with 1. degree taylor approximation
            # Taylor approximation: np.sqrt(R**2 - a**2) + R - (R/np.sqrt(R**2 - a**2))*(x-a)
            # Cut circle in upper (1) and lower (-1) part
            sn = (1 if i > NUM_WP/4 and i < NUM_WP*3/4 else -1)
            NST = (R*np.sqrt(R**2 - xi**2)-sn*xi**2+xi*R+sn*R**2)/R
            h = np.sqrt(yi**2 + (xi-NST)**2)
            # Top right and bottom left part of circle
            if (i > NUM_WP/4 -1 and i < NUM_WP*2/4 -1) or (i > NUM_WP*3/4 -1 and i < NUM_WP -1):
                TARGET_RPY[i, :] = 0, 0, -(np.pi - np.arcsin(yi/h))
            # Bottom right and top left part of circle
            if (i > 0 and i < NUM_WP/4 -1) or (i > NUM_WP*2/4 -1 and i < NUM_WP*3/4 -1):
                TARGET_RPY[i, :] = 0, 0, -np.arcsin(yi/h)

        print(f"MATRIX OF YAW ANGLES FOR CIRCLE FLYING: {TARGET_RPY}")

        wp_counter = 0

    if traj == 'loop': #Play for 15 sec with --duration_sec 15
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP,3))

        for i in range(NUM_WP):
            TARGET_POS[i, :] = (R*np.cos((i/NUM_WP)*(2*np.pi)-np.pi/2)+INIT_XYZS[0, 0]), INIT_XYZS[0, 1], INIT_XYZS[0, 2]-(R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R)
        wp_counter = 0

    # TODO
    if traj == 'forwardloop': #Play for 15 sec with --duration_sec 15
        PERIOD = 10
        NUM_WP = ARGS.control_freq_hz*PERIOD
        TARGET_POS = np.zeros((NUM_WP+2,3))

        TARGET_POS[0, :] = INIT_XYZS[0, 0] + 1, INIT_XYZS[0, 1], INIT_XYZS[0, 2]
        TARGET_POS[NUM_WP+2-1, :] = INIT_XYZS[0, 0] + 2, INIT_XYZS[0, 1], INIT_XYZS[0, 2]

        for i in range(NUM_WP):
            TARGET_POS[i+1, :] = (R*np.cos((i/NUM_WP)*(2*np.pi)-np.pi/2)+TARGET_POS[0, 0]), TARGET_POS[0, 1], TARGET_POS[0, 2]-(R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R)
        wp_counter = 0

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(env) for i in range(ARGS.num_drones)]
    # ctrl = [SimplePIDControl(env) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    reward = 0
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, _, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action[str(0)], _, _ = ctrl[0].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=obs[str(0)]["state"],
                                                                    target_pos=TARGET_POS[wp_counter, :].reshape((3,)),
                                                                    target_rpy=TARGET_RPY[wp_counter, :].reshape((3,)) if traj=='ascent_rpy' else None
                                                                    )


            #### Go to the next way point and loop #####################
            wp_counter = wp_counter + (1 if wp_counter < (NUM_WP-1) else 0)
            # Why are there wp counters? -> to know which drone is at which waypoint

        #### Log the simulation ####################################
        logger.log(drone=0,
                    timestamp=i/env.SIM_FREQ,
                    state= obs[str(0)]["state"],
                    control=np.hstack([TARGET_POS[wp_counter, :].reshape((3,)), (TARGET_RPY[wp_counter, :].reshape((3,)) if traj == 'ascent_rpy' else np.zeros(3)), np.zeros(6)])
                    )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                print(obs[str(0)]["rgb"].shape, np.average(obs[str(0)]["rgb"]),
                        obs[str(0)]["dep"].shape, np.average(obs[str(0)]["dep"]),
                        obs[str(0)]["seg"].shape, np.average(obs[str(0)]["seg"])
                        )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

        #### Calculate the reward ##################################
        # TODO: Reward function seems weird, no penalty if far away from target
        state = obs[str(0)]["state"]
        if traj == 'hover':
            reward += -1 * np.maximum(0, 1-np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2)
        if traj == 'forward':
            reward += -1 * np.maximum(0, 1-np.linalg.norm(np.array([1, 0, 0.5])-state[0:3])**2)
        if traj == 'turns':
            reward += -1 * np.maximum(0, 1-np.linalg.norm(np.array([8*math.pi, 0, 0.5])-state[0:3])**2)
        if traj == 'circle':
            reward += -1 * np.maximum(0, 1-np.linalg.norm(INIT_XYZS-state[0:3])**2)
        if traj == 'ascent' or traj == 'ascent_rpy':
            reward += -1 * np.maximum(0, 1-np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2)
        if traj == 'loop':
            reward = 0

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Print the reward ######################################
    print(f"Reward of drone when using PID control for {traj}: {reward}\n")

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()


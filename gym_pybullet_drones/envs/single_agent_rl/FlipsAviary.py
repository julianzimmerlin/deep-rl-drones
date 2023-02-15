import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

class FlipsAviary(BaseSingleAgentAviary):
    """Single agent RL problem: Flips at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 state_prev=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 use_advanced_loss=False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=np.array([[0, 0, 1]]),
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        print('[FlipsAviary]: Using advanced loss: '+str(use_advanced_loss))
        self.use_advanced_loss = use_advanced_loss
        # self.initial_xyzs = (initial_xyzs if initial_xyzs != None else np.array([[0, 0, 1]]))
        self.initial_xyzs = np.array([[0, 0, 1]])
        # Class variables
        self.state_prev = self._getDroneStateVector(0)
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        R = 0.3 # Flips radius
        PERIOD = 10

        # Normalization bounds, see _clipAndNormalizeState method
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range
        MAX_PITCH_ROLL_VEL = 6*np.pi
        MAX_YAW_VEL = 3*np.pi

        state = self._getDroneStateVector(0)
        rand = np.random.randint(0, 101)
        
        if self.use_advanced_loss:  # state[7:10] are RPY angles and state[13:16] are angular velocities
            
            # Normalization to (0, upper_bound) with X_changed = (X-X_min)/(X_max - X_min) * upper_bound
            # https://inomics.com/blog/standardizing-the-data-of-different-scales-which-method-to-use-1036202
            # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
            position_state = (state[0:2] - (-MAX_XY))/(MAX_XY - (-MAX_XY))
            position_initial = (self.initial_xyzs[0,0:2] - (-MAX_XY))/(MAX_XY - (-MAX_XY))
            position_z_state = (state[2] - 0)/(MAX_Z - 0)
            position_z_initial = (self.initial_xyzs[0,2] - 0)/(MAX_Z - 0)
            angle_state = (state[8:10] - (-MAX_PITCH_ROLL))/(MAX_PITCH_ROLL - (-MAX_PITCH_ROLL))
            roll_state = (state[7] - (-MAX_PITCH_ROLL))/(MAX_PITCH_ROLL - (-MAX_PITCH_ROLL))
            roll_prev = (self.state_prev[7] - (-MAX_PITCH_ROLL))/(MAX_PITCH_ROLL - (-MAX_PITCH_ROLL))
            angular_v_state = (state[13] - (-MAX_PITCH_ROLL_VEL))/(MAX_PITCH_ROLL_VEL - (-MAX_PITCH_ROLL_VEL))
            vel_state = (state[10:13] - (-MAX_LIN_VEL_XY))/(MAX_LIN_VEL_XY - (-MAX_LIN_VEL_XY))
            
            position_loss = np.linalg.norm(position_initial-position_state)**2
            position_z_loss = np.maximum(0, position_z_state-position_z_initial) # slightly reward going up a bit
            angle_loss = np.linalg.norm(angle_state)
            roll_angle_loss = roll_state
            roll_change = roll_state-roll_prev
            angular_v_loss = angular_v_state # reward roll angular vel
            vel_loss = np.linalg.norm(vel_state)

            # Clipping to range of (0,1) after normalizing            
            if rand >= 99:   
                print (f"DEBUGGING INFORMATION: after normalization \nPosition Loss: {position_loss} \nHeight Loss: {position_z_loss} \nAngle Loss: {angle_loss} \nRoll Loss: {angular_v_loss} \nRoll Angle Loss: {roll_angle_loss} \nRoll Change: {roll_change} \nVel Loss: {vel_loss} \n")

            # 1- means we "penalize", which means we reward undesired actions less
            position_loss = 1-np.maximum(0, np.minimum(position_loss, 1))
            position_z_loss = np.maximum(0, np.minimum(position_z_loss, 1))
            angle_loss = 1-np.maximum(0, np.minimum(angle_loss, 1))
            roll_angle_loss = np.maximum(0, np.minimum(roll_angle_loss, 1))
            roll_change = np.maximum(0, np.minimum(roll_change, 1))
            angular_v_loss = np.maximum(0, np.minimum(angular_v_loss, 1))
            vel_loss = 1-np.maximum(0, np.minimum(vel_loss, 1))

            if rand >= 99:   
                print (f"DEBUGGING INFORMATION: after clipping \nPosition Loss: {position_loss} \nHeight Loss: {position_z_loss} \nAngle Loss: {angle_loss} \nRoll Loss: {angular_v_loss} \nRoll Angle Loss: {roll_angle_loss} \nRoll Change: {roll_change} \nVel Loss: {vel_loss} \n")

            # Factors
            roll_fac = 1
            z_fac = 0.05
            vel_fac = 0.1
            ang_fac = 0.1
            pos_fac = 0.1

            # Save state as previous state
            self.state_prev = state

            if rand >= 99:
                #print (f"DEBUGGING INFORMATION: after weighting \nPosition Loss: {pos_fac*(1/7)*position_loss} \nHeight Loss: {z_fac*(1/7)*position_z_loss} \nAngle Loss: {ang_fac*(1/7)*angle_loss} \nRoll Loss: {roll_fac*(1/7)*angular_v_loss} \nRoll Angle Loss: {roll_fac*(1/7)*roll_angle_loss} \nRoll Change: {roll_fac*(1/7)*roll_change} \nVel Loss: {vel_fac*(1/7)*vel_loss} \n")
                print (f"DEBUGGING INFORMATION: after weighting \nPosition Loss: {pos_fac*(1/4)*position_loss} \nHeight Loss: {z_fac*(1/4)*position_z_loss} \nAngle Loss: None \nRoll Loss: None \nRoll Angle Loss: None \nRoll Change: {roll_fac*(1/4)*roll_change} \nVel Loss: {vel_fac*(1/4)*vel_loss} \n")
                print(f"DEBUGGING INFORMATION: \nReward: {pos_fac*(1/4)*position_loss + z_fac*(1/4)*position_z_loss + vel_fac*(1/4)*vel_loss + roll_fac*(1/4)*roll_change} \n")
            return pos_fac*(1/4)*position_loss + z_fac*(1/4)*position_z_loss + vel_fac*(1/4)*vel_loss + roll_fac*(1/4)*roll_change # (0, 1)
            #return pos_fac*(1/7)*position_loss + z_fac*(1/7)*position_z_loss + vel_fac*(1/7)*vel_loss + roll_fac*(1/7)*angular_v_loss + ang_fac*(1/7)*angle_loss + roll_fac*(1/7)*roll_angle_loss + roll_fac*(1/7)*roll_change # (0, 1)
        else:
            roll_change = state[7]-self.state_prev[7]
            # Save state as previous state
            self.state_prev = state
            return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2 + roll_change
        # Loss clipping: clipped_loss = max(0, min(loss, upper_bound))

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range
        MAX_PITCH_ROLL_VEL = 6*np.pi
        MAX_YAW_VEL = 3*np.pi

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        clipped_ang_vel_rp = np.clip(state[13:15], -MAX_PITCH_ROLL_VEL, MAX_PITCH_ROLL_VEL)
        clipped_ang_vel_y = np.clip(state[15], -MAX_YAW_VEL, MAX_YAW_VEL)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z,
                                               clipped_ang_vel_rp,
                                               clipped_ang_vel_y
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel_rp = clipped_ang_vel_rp / MAX_PITCH_ROLL_VEL
        normalized_ang_vel_y = clipped_ang_vel_y / MAX_YAW_VEL

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel_rp,
                                      normalized_ang_vel_y,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      clipped_ang_vel_rp,
                                      clipped_ang_vel_y
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if any of the 20 values in a state vector is out of the normalization range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
        if not(clipped_ang_vel_rp == np.array(state[13:15])).all():
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped angular velocity [{:.2f} {:.2f}]".format(state[13], state[14]))
        if not(clipped_ang_vel_y == np.array(state[15])):
            print("[WARNING] it", self.step_counter, "in FlipsAviary._clipAndNormalizeState(), clipped angular velocity [{:.2f}]".format(state[15]))

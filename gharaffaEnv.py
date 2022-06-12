## Custom GymEnv, copied from Stefano's code

import logging
import os
import sys
from time import sleep
import string
import random
import time

import numpy as np

import gym
from gym import spaces,error
from gym.utils import seeding

import traci

# import the base environment class
#from flow.envs import Env

# sumo config files are housed here
# TODO: Find a way to integrate them into Flow using simulator-agnostic APIs as much as possible
module_path = './TrafQ/Environments/gym-gharrafa/gymGharrafa'

# set up sumo home
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
sumoBinary = "sumo"
sumoBinaryGUI = "sumo-gui"

# define the environment class, and inherit properties from the base environment class
class gharaffaEnv(gym.Env):
        
    def __init__(self, env_config):

        self.tlsID = "6317"
        self._seed = 31337
        self.seed(31337)

        self.GUI = True
        if 'GUI' in env_config:
            self.GUI = env_config['GUI']
        self.Play = None
        if 'Play' in env_config:
            self.Play = env_config['Play']
        self.Mode = "train" #If train, use default episodic conditions; If eval, run full hour 
        if 'Mode' in env_config:
            self.Mode = env_config['Mode']
        self.s_hour = 7
        self.e_hour = 8
        self.day = 0
        if 'sHour' in env_config:
            self.s_hour = env_config['sHour']
        if 'eHour' in env_config:
            self.e_hour = env_config['eHour']
        if 'day' in env_config:
            self.day = env_config['day']


        self.PHASES = {
        0: "G E",
        1: "G N",
        2: "G W",
        3: "G S",
        4: "ALL RED",
        5: "G EW",
        6: "G NS",
        7: "G E BY W",
        8: "G N BY S",
        9: "G S BY N",
        10: "G W BY E"
        }

        self.monitored_edges = ["7552",
        "7553",
        "7554",
        "7556",
        "7558",
        "7560",
        "7562",
        "7563",
        "7565",
        "7574",
        "6043",
        "7593",
        "7594",
        "7621",
        "7623",
        "10324",
        "10339",
        "6124",
        "7665",
        "7542",
        "7673",
        "7547",
        "7548",
        "7549",
        "7550"]

        self.DETECTORS = []
        with open(module_path +"/assets/gharrafa_detectors") as fdet:
            for detname in fdet:
                self.DETECTORS.append(detname[:-1])

        #how many SUMO seconds before next step (observation/action)
        self.OBSERVEDPERIOD = 10
        self.SUMOSTEP = 0.5

        #In this version the observation space is the set of sensors
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,68), dtype=np.uint8)

        #Set action space as the set of possible phases
        self.action_space = spaces.Discrete(11)


        #Generate an alphanumerical code for the run label (to run multiple simulations in parallel)
        self.runcode = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

        self.timestep = 0

        self._configure_environment()

    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _configure_environment(self):

        if self.GUI:
            sumoBin = sumoBinaryGUI
        else:
            sumoBin = sumoBinary

        self.argslist = [sumoBin, "-n", module_path+"/assets/gharrafa.net.xml",
            "-r", module_path+"/assets/MATOD/gha_trips_"+str(self.s_hour)+"-"+str(self.e_hour)+"_R"+str(self.day)+".rou.xml",
            "-a", module_path+"/assets/TAZ/gharrafa_taz.xml, "+module_path+"/assets/e1test.xml",
            "-g", module_path+"/assets/gui.settings.xml",
            "-b", str(3600*self.s_hour), "-e", str(3600*self.e_hour), # begin time and end time in seconds
            "--device.rerouting.with-taz",
            "--collision.action", "remove", "--scale", "1",
            "--step-length", str(self.SUMOSTEP), "-S", "--time-to-teleport", "-1",
            "--collision.mingap-factor", "0", "--ignore-junction-blocker", "3",
            "--collision.check-junctions", "true", "--no-step-log", "--no-warnings"]

        if self.Play:
            #self.argslist.append("--game")
            self.argslist.append("--window-size")
            self.argslist.append("1000,1000")

        # if self.GUI:
        #     self.arglist.append("--gui-settings-file")
        #     self.arglist.append(module_path+"/assets/viewsettings.xml")

        traci.start(self.argslist,label=self.runcode)

        self.conn = traci.getConnection(self.runcode)

        time.sleep(1) # Wait for server to startup

    def __del__(self):
        self.conn.close()

    def closeconn(self):
        self.__del__()

    def _selectPhase(self,target):
        target = self.PHASES[target]
        current_program = self.conn.trafficlight.getProgram(self.tlsID)
        if " to " in current_program:
            source = current_program.split(" to ")[1]
        else:
            #we are in another program like scat or 0... just change from N
            source = "G N"
            if source == target:
                source = "G S"
        if source == target and " to " in current_program:
            #ensure that the phase will not be changed automatically by the
            #program, by adding some time
            self.conn.trafficlight.setPhase(self.tlsID, 1)
            self.conn.trafficlight.setPhaseDuration(self.tlsID,60.0)
            return False
        else:
            transition_program = "from %s to %s" % (source,target)
            self.conn.trafficlight.setProgram(self.tlsID, transition_program)
            self.conn.trafficlight.setPhase(self.tlsID, 0)
            return True

    def _observeState(self):
        #selftimestep = self.conn.simulation.getCurrentTime()/1000
        lastVehiclesVector = np.zeros(len(self.DETECTORS),dtype=np.float32)
        reward = 0

        #initialize accumulators for Play mode measures
        ACCgetWaitingTime = 0
        ACCgetTravelTime = 0

        ACCgetLastStepOccupancy = 0
        ACCgetLastStepMeanSpeed = 0
        ACCgetLastStepHaltingNumber = 0

        ACCgetCO2Emission = 0
        ACCgetNOxEmission = 0
        ACCgetHCEmission = 0
        ACCgetNoiseEmission = 0

        ACCgetArrivedNumber = 0
        ACCgetDepartedNumber = 0
        #ACCgetCollidingVehiclesNumber = 0

        measures = {}


        for i in range(int(self.OBSERVEDPERIOD/self.SUMOSTEP)):
            self.conn.simulationStep()
            self.timestep += self.SUMOSTEP #self.conn.simulation.getCurrentTime()/1000
            lastVehiclesVector += np.array([np.float32(self.conn.inductionloop.getLastStepVehicleNumber(detID)) for detID in self.DETECTORS])
            reward += np.sum([self.conn.inductionloop.getLastStepVehicleNumber(detID) for detID in self.DETECTORS if "out_for" in detID])
            if self.Play != None:
                #measure delay,emissions etc. from selected edges
                ACCgetWaitingTime += np.mean([self.conn.edge.getWaitingTime(edgeID) for edgeID in self.monitored_edges])  #must average over micro steps
                ACCgetTravelTime += np.mean([self.conn.edge.getTraveltime(edgeID) for edgeID in self.monitored_edges])  #must average over micro steps

                ACCgetLastStepOccupancy += np.mean([self.conn.edge.getLastStepOccupancy(edgeID) for edgeID in self.monitored_edges])  # must average over micro steps
                ACCgetLastStepMeanSpeed += np.mean([self.conn.edge.getLastStepMeanSpeed(edgeID) for edgeID in self.monitored_edges])  # must average over micro steps
                ACCgetLastStepHaltingNumber += np.mean([self.conn.edge.getLastStepHaltingNumber(edgeID) for edgeID in self.monitored_edges])  # must average over micro steps

                ACCgetCO2Emission += np.sum([self.conn.edge.getCO2Emission(edgeID) for edgeID in self.monitored_edges])  # must sum over micro steps
                ACCgetNOxEmission += np.sum([self.conn.edge.getNOxEmission(edgeID) for edgeID in self.monitored_edges])  # must sum over micro steps
                ACCgetHCEmission += np.sum([self.conn.edge.getHCEmission(edgeID) for edgeID in self.monitored_edges])  # must sum over micro steps
                ACCgetNoiseEmission += np.sum([self.conn.edge.getNoiseEmission(edgeID) for edgeID in self.monitored_edges])  # must sum over micro steps

                ACCgetArrivedNumber += self.conn.simulation.getArrivedNumber()  # must sum over micro steps
                ACCgetDepartedNumber += self.conn.simulation.getDepartedNumber()  # must sum over micro steps
                #ACCgetCollidingVehiclesNumber += self.conn.simulation.getCollidingVehiclesNumber()  # must sum over micro steps

        measures = {
        "WaitingTime" : ACCgetWaitingTime/(i+1),
        "TravelTime" : ACCgetTravelTime/(i+1),

        "Occupancy" : ACCgetLastStepOccupancy/(i+1),
        "MeanSpeed" : ACCgetLastStepMeanSpeed/(i+1),
        "HaltingNumber" : ACCgetLastStepHaltingNumber/(i+1),

        "CO2Emission" : ACCgetCO2Emission,
        "NOxEmission" : ACCgetNOxEmission,
        "HCEmission" : ACCgetHCEmission,
        "NoiseEmission" : ACCgetNoiseEmission,

        "ArrivedNumber" : ACCgetArrivedNumber,
        "DepartedNumber" : ACCgetDepartedNumber,

        "Reward" : reward#,
        #"CollidingVehiclesNumber" : ACCgetCollidingVehiclesNumber
        }

        obs = lastVehiclesVector

        #TODO: build observation
        return obs,float(reward),measures

    def step(self, action):
        if self.Play != None and self.Play != "action":
            obs,reward,measures = self._observeState()
            measures["time"] = self.timestep

            #episodic conditions
            done = self.timestep >= 3600
            if self.Mode != 'eval': # no episodic conditions
                c1 = self.conn.lane.getLastStepHaltingNumber("7594_2")>10
                c2 = self.conn.lane.getLastStepHaltingNumber("6511_1")>10
                c3 = self.conn.lane.getLastStepHaltingNumber("7673_0")>10
                done |= (c1 or c2 or c3)

            if done:
                self.timestep = 0
                #time.sleep(1.0)
                self.conn.load(self.argslist[1:])
            return obs, reward, done, measures

        episode_over=False
        self._selectPhase(action)

        #get state and reward
        obs,reward,measures = self._observeState()
        #print(f'For step: {action}, observed: {obs}, reward: {reward}, measures: {measures}')

        measures["time"] = self.timestep

        if self.Mode == 'eval': # no episodic conditions
            if self.timestep >= 3600: # corresponds to 1 hr
                #print(f'Episode in evaluation finished 3600 timesteps')
                episode_over = True
                self.timestep = 0
                #time.sleep(1.0)
                self.conn.load(self.argslist[1:])
                #time.sleep(1.0)

            return obs, reward, episode_over, measures
            
        #episodic conditions
        c1 = self.conn.lane.getLastStepHaltingNumber("7594_2")>10
        c2 = self.conn.lane.getLastStepHaltingNumber("6511_1")>10
        c3 = self.conn.lane.getLastStepHaltingNumber("7673_0")>10
        #detect "game over" state
        if self.Play != "action" and (self.timestep >= 3600 or c1 or c2 or c3):
        #if self.timestep >= 3600 or c1 or c2 or c3:
            episode_over = True
            self.timestep = 0
            #time.sleep(1.0)
            self.conn.load(self.argslist[1:])
            #time.sleep(1.0)
            #copyfile("/home/srizzo/tls_switch_states.xml","/home/srizzo/phase_recording/%s_tls_switch_states_%d.xml" % (self.runcode,self.timestep))
            #self.conn.gui.screenshot(viewID='View #0',filename="/home/srizzo/phase_recording/%s_last_screen.png" % self.runcode)

        if self.Play == "action" and self.GUI:
            self.conn.gui.screenshot(viewID='View #0',filename="/tmp/phase_recording/%s_last_screen.png" % self.runcode)

        if self.Play == "action" and (self.timestep >= 3600 or c1 or c2 or c3):
            episode_over = True
            #copyfile("/home/srizzo/tls_switch_states.xml","/home/srizzo/phase_recording/%s_tls_switch_states_%d.xml" % (self.runcode,self.timestep))


        return obs, reward, episode_over, measures

    def reset(self, state_only=False):
        self.timestep = 0
        ## pick random traffic for this episode
        if not state_only:
            self.s_hour = self.np_random.randint(0, 24)
            self.e_hour = self.s_hour + 1
            self.day = self.np_random.randint(0, 7)
            print(f'Reset start hour to {self.s_hour} and day to {self.day}')
            self.closeconn()
            #Generate an alphanumerical code for the run label (to run multiple simulations in parallel)
            self.runcode = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self._configure_environment()

        #go back to the first step of the return
        if self.Play != None and self.Play != "action":
            self.conn.trafficlight.setProgram(self.tlsID, self.Play)
        if self.Play == "action" and self.GUI:
            self.conn.gui.screenshot(viewID='View #0',filename="/tmp/phase_recording/%s_last_screen.png" % self.runcode)

        return self._observeState()[0]

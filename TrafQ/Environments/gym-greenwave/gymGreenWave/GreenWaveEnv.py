from .assets import set_sumo_home

import logging
import os
from time import sleep
import string
import random
import time

from shutil import copyfile

import numpy as np

import gym
from gym import spaces,error
from gym.utils import seeding

import traci

module_path = os.path.dirname(__file__)

class GreenWaveEnv(gym.Env):
    def __init__(self,oneway=True,uneven=False,GUI=True,minlength=1,macrostep=1,minsteps=1):

        self.GUI = GUI
        self.tls = {
        0:"gneJ37",
        1:"gneJ38",
        2:"gneJ39",
        3:"gneJ40"
        }

        self.steps_since_last_change = len(self.tls.keys()) * [1]

        #used to force waiting in same phase if has not lasted enough, the action has no effect
        self.minlength = minlength

        #used to force waiting in same phase if has not lasted enough, the action is taken after
        #at least minsteps are past in the last phase
        self.minsteps = minsteps

        #used to make x micro steps in sumo, while it is only one step from the agent point of view
        self.macrostep=macrostep

        self._seed = 31337

        self.PHASES = [0,1]

        #how many SUMO seconds before next step (observation/action)
        self.OBSERVEDPERIOD = 10
        self.SUMOSTEP = 1.0

        #In this version the observation space is the set of sensors
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,24), dtype=np.uint8)

        self.DETECTORS = ["e2Detector_--gneE13_0_14",
        "e2Detector_--gneE14_0_17",
        "e2Detector_--gneE15_0_20",
        "e2Detector_--gneE16_0_23",
        "e2Detector_-gneE13_0_4",
        "e2Detector_-gneE14_0_7",
        "e2Detector_-gneE15_0_9",
        "e2Detector_-gneE16_0_11",
        "e2Detector_gneE13_0_5",
        "e2Detector_gneE14_0_6",
        "e2Detector_gneE15_0_8",
        "e2Detector_gneE16_0_10",
        "e2Detector_gneE20_0_0",
        "e2Detector_gneE21_0_1",
        "e2Detector_gneE21_0_13",
        "e2Detector_gneE22_0_16",
        "e2Detector_gneE22_0_2",
        "e2Detector_gneE23_0_19",
        "e2Detector_gneE23_0_3",
        "e2Detector_gneE24_0_22",
        "e2Detector_gneE35_0_12",
        "e2Detector_gneE38_0_15",
        "e2Detector_gneE39_0_22",
        "e2Detector_gneE41_0_23"]

        self.intersections = [0,1,2,3]
        # DETDICT FORMAT EXAMPLE: self.DETDICT[(IntersectionID,DirectionID)]["in"] = ["e2Detector_gneE23_0_3","e2Detector_gneE24_0_22"]
        self.DETDICT = {
        (0,0): {"in":["e2Detector_gneE20_0_0"],"out":["e2Detector_gneE21_0_13"]},
        (1,0): {"in":["e2Detector_gneE21_0_1"],"out":["e2Detector_gneE22_0_16"]},
        (2,0): {"in":["e2Detector_gneE22_0_2"],"out":["e2Detector_gneE23_0_19"]},
        (3,0): {"in":["e2Detector_gneE23_0_3"],"out":["e2Detector_gneE24_0_22"]},

        (0,1): {"in":["e2Detector_gneE13_0_5"],"out":["e2Detector_--gneE13_0_14"]},
        (1,1): {"in":["e2Detector_gneE14_0_6"],"out":["e2Detector_--gneE14_0_17"]},
        (2,1): {"in":["e2Detector_gneE15_0_8"],"out":["e2Detector_--gneE15_0_20"]},
        (3,1): {"in":["e2Detector_gneE16_0_10"],"out":["e2Detector_--gneE16_0_23"]},

        (0,2): {"in":["e2Detector_-gneE13_0_4"],"out":["e2Detector_gneE35_0_12"]},
        (1,2): {"in":["e2Detector_-gneE14_0_7"],"out":["e2Detector_gneE38_0_15"]},
        (2,2): {"in":["e2Detector_-gneE15_0_9"],"out":["e2Detector_gneE39_0_22"]},
        (3,2): {"in":["e2Detector_-gneE16_0_11"],"out":["e2Detector_gneE41_0_23"]}
        }

        self.EDGESmain = ["gneE20","gneE21","gneE22","gneE23","gneE24"]

        self.EDGES = []

        #Set action space as the set of possible phases
        self.action_space = spaces.Discrete(16)

        self.spec = {}

        #Generate an alphanumerical code for the run label (to run multiple simulations in parallel)
        self.runcode = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

        self.timestep = 0

        self._configure_environment()

    def _getPressure(self,IntersectionID):
        directions = [k[1] for k in self.DETDICT.keys() if k[0]==IntersectionID]

        pressure = 0.0
        for direction in directions:
            inDets = self.DETDICT[IntersectionID,direction]["in"]
            inPressure = np.sum([ self.conn.lanearea.getLastStepVehicleNumber(detID) for detID in inDets])

            outDets = self.DETDICT[IntersectionID,direction]["out"]
            outPressure = np.sum([ self.conn.lanearea.getLastStepVehicleNumber(detID) for detID in outDets])

            pressure += inPressure - outPressure

        return pressure


    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _configure_environment(self):
        if self.GUI:
            sumoBinary = set_sumo_home.sumoBinaryGUI
        else:
            sumoBinary = set_sumo_home.sumoBinary

        self.argslist = [sumoBinary, "-c", module_path+"/assets/greenWave.sumocfg",
                             "--collision.action", "teleport",
            "--step-length", str(self.SUMOSTEP), "-S", "--time-to-teleport", "-1",
            "--collision.mingap-factor", "0",
            "--collision.check-junctions", "true", "--no-step-log"]

        traci.start(self.argslist,label=self.runcode)

        self.conn = traci.getConnection(self.runcode)

        time.sleep(5) # Wait for server to startup

        #get list of all edges
        self.EDGES = self.conn.edge.getIDList()

    def __del__(self):
        self.conn.close()

    def closeconn(self):
        self.conn.close()

    def _getActionFromPhase(self,phase):
        #if the tl is transitioning consider the next phase
        return {0:0, 1:1, 2:1, 3:0}[phase]

    def _selectPhase(self,targets):
        for i,action in enumerate(targets):
            lastPhase = self.conn.trafficlight.getPhase(self.tls[i])
            lastAction = self._getActionFromPhase(lastPhase)
            if action != lastAction and self.steps_since_last_change[i]>= self.minlength:
                while self.steps_since_last_change[i] < self.minsteps:
                    #if min steps requirement is set, do other steps until it's satisfied
                    self.conn.simulationStep()
                    self.steps_since_last_change = [x+1 for x in self.steps_since_last_change]
                    self.timestep +=1
                self.conn.trafficlight.setPhase(self.tls[i],(lastPhase+1)%4)
                self.steps_since_last_change[i]=0
            elif lastPhase%2==0:
                #extend the phase if it's not yellow (the length in tls definition may not be long
                #enough, thus it may skip to another phase by itself otherwise)
                self.conn.trafficlight.setPhaseDuration(self.tls[i],60.0)


    def _observeState(self):
        reward = [0.0 for i in self.intersections]

        #multi agent observation
        observations = []
        for i in self.intersections:
            idets = []
            for k in self.DETDICT.keys():
                if k[0] == i:
                    idets+=self.DETDICT[k]["in"]
                    idets+=self.DETDICT[k]["out"]
            #dets = [self.DETDICT[k]["in"] for k in self.DETDICT.keys() if k[0] == i]
            #dets += [self.DETDICT[k]["out"] for k in self.DETDICT.keys() if k[0] == i]
            obs = [np.float32(self.conn.lanearea.getLastStepVehicleNumber(detID)) for detID in idets ]
            lastPhase = self.conn.trafficlight.getPhase(self.tls[i])
            lastAction = self._getActionFromPhase(lastPhase)
            obs.append(lastAction)
            observations.append(obs)


        #number = [np.float32(self.conn.lanearea.getLastStepVehicleNumber(detID)) for detID in self.DETECTORS]
        #speed = [np.float32(self.conn.lanearea.getLastStepMeanSpeed(detID)) for detID in self.DETECTORS]
        #halting = [np.float32(self.conn.lanearea.getLastStepHaltingNumber(detID)) for detID in self.DETECTORS]
        #obs = np.array(number + speed + halting)
        #obs = np.array(number)

        # Note: edge.getWaitingTime(edgeID) Returns the sum of the waiting time of all vehicles currently
        # on that edge edgeID
        #waitingALL = np.sum([self.conn.edge.getWaitingTime(edgeID) for edgeID in self.EDGES])
        #waitingMain = np.sum([self.conn.edge.getWaitingTime(edgeID) for edgeID in self.EDGESmain])
        #reward = -np.log(waitingMain) if waitingMain > 0 else 0

        #In multi-agent with shared reward we should use sum of all the local rewards
        #totalpressure = np.sum([getpressure(i) for i in self.intersections])
        #reward = -totalpressure
        reward = [-self._getPressure(i) for i in self.intersections]

        measures = {}
        #TODO: build observation
        return observations,reward,measures

    def step(self, action):
        self._selectPhase(action)
        #self.conn.simulation.step(time=10.0)
        for i in range(self.macrostep):
            self.conn.simulationStep()
            self.steps_since_last_change = [x+1 for x in self.steps_since_last_change]
            self.timestep +=1
        #get state and reward
        obs,reward,measures = self._observeState()
        episode_over = self.timestep >= (360000-1) or np.sum(reward) < -16
        if episode_over:
            self.conn.load(self.argslist[1:])
            self.timestep = 0
        return obs, reward, episode_over, measures

    def reset(self):
        self.timestep = 0
        return self._observeState()[0]

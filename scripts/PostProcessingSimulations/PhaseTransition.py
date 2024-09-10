import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from DateTimeHandler import *
sys.path.append(os.path.join(os.environ["TRAFFIC_DIR"],"scripts"))
from FittingProcedures import *

def AnalyzePeopleInNetwork(Rs,TimeInterval2NumberPeopleInNet):
    R2Time2ErrorFit = {R:{} for R in Rs}
    for R in Rs:
        R2Time2ErrorFit[R],Tau = ComputeAlpha(TimeInterval2NumberPeopleInNet)

def ComputeAlpha(TimeInterval2NumberPeopleInNet):
    """
        Description:
            - Computes the Alpha for the Power Law
        Args:
            - TimeInterval2NumberPeopleInNet: dict -> {time_interval:number_people}
        Returns:
            - Alpha: float
        NOTE: For a fixed R
        TODO: 
            Define the number of people in time. (Done)
            Define the window to compute the fit. (Done)
            Choose the index in time that corresponds to the best fit. (Done)
            Take it as N(tau) = N(R)
    """
    StartTime = 7
    StartingBin = int(StartTime*MINUTES_IN_HOUR/15)
    NumIntervals = len(TimeInterval2NumberPeopleInNet.keys()) - StartingBin
    Hours = list(TimeInterval2NumberPeopleInNet.keys())[StartingBin:]
    IntTimeArray = np.linspace(0,HOURS_IN_DAY,NumIntervals,dtype = float)[StartingBin:]       
    NPeople = np.array(list(TimeInterval2NumberPeopleInNet.values()))[StartingBin:]
    print("Number Intervals: ",NumIntervals)
    print("Hours: ",Hours)
    print("IntTimeArray: ",IntTimeArray)
    print("NPeople: ",NPeople)
    # NOTE: Important to Normalize to to Compare Power Law and Expo with the right Initial Guess
    PPeople = NPeople/np.sum(NPeople)
    # Time Windows for which the 
    Windows = {t:IntTimeArray[StartingBin:StartingBin + t] for t in range(4,len(IntTimeArray))}
    print("Windows: ",Windows)
    print("len Windows: ",len(Windows)," len PPeople: ",len(PPeople)," len Hours: ",len(Hours)," len IntTimeArray: ",len(IntTimeArray))
    # Time Error Fit
    Time2ErrorFit = {t:{'error':0,"A":0,"b":0,"is_powerlaw":False} for t in range(1,len(Hours))}
    for windowTime in Windows.keys():
        x = np.array(Windows[windowTime][:windowTime])
        y = np.array(PPeople[:windowTime])
        Time2ErrorFit[windowTime]['error'],Time2ErrorFit[windowTime]['A'],Time2ErrorFit[windowTime]['b'],Time2ErrorFit[windowTime]['is_powerlaw'] = ComparePlExpo(x,y)
        print("Window Time: ",windowTime)
        print(Time2ErrorFit[windowTime]['error'])
        print('A: ',Time2ErrorFit[windowTime]['A'])
        print('b: ',Time2ErrorFit[windowTime]['b'])
        print('Is Power Law: ',Time2ErrorFit[windowTime]['is_powerlaw'])
    # Get the best Tau
    Tau = GetTauForNR(Time2ErrorFit)
    print("Tau: ",Tau)
    print(Time2ErrorFit)
    return Time2ErrorFit,Tau
    

def GetTauForNR(Time2ErrorFit):
    """
        Chooses the window for Which The Power Law is best fitted.
        NOTE: Given R
    """
    SmallestErrorWindow = 1000000
    for TimeWindow in Time2ErrorFit.keys():
        if Time2ErrorFit[TimeWindow]['is_powerlaw']:
            if Time2ErrorFit[TimeWindow]['error']<SmallestErrorWindow:
                SmallestErrorWindow = Time2ErrorFit[TimeWindow]['error']
                BestTimeWindow = TimeWindow
            else:
                pass
    return BestTimeWindow



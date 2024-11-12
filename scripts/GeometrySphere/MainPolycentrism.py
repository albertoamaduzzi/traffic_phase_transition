import numpy as np
import json
import os
import socket
from Potential import *
from ModifyPotential import *
from Polycentrism import *
from PolycentrismPlot import *
from GenerateModifiedFluxesSimulation import *
from ODfromfma import *
import logging
logger = logging.getLogger(__name__)

# ----- UPLOAD GRAVITATIONAL FIT ------
def UploadGravitationalFit(TRAFFIC_DIR,name):
    with open(os.path.join(TRAFFIC_DIR,'data','carto',name,'potential','FitVespignani.json'),'r')as f:
        fitGLM = json.load(f)
    k = np.exp(fitGLM['logk'])
    alpha =fitGLM['alpha']
    beta = fitGLM['gamma']
    d0 = fitGLM['1/d0']    
    return k,alpha,beta,d0


def ConcatenateODsInputSimulation(SFO_obj,grid_size,NameCity):
    FirstFile = True
    for file in os.listdir(os.path.join(SFO_obj.ODfma_dir)):
        if file.endswith('.fma'):
            start = int(file.split('.')[0].split('D')[1])
            end = start + 1
            ODfmaFile = os.path.join(SFO_obj.ODfma_dir,file)
            if start == 7:
                pass
            else:
                if FirstFile:
                    df1,_,ROutput = OD_from_fma(SFO_obj.polygon2OD,
                                        SFO_obj.osmid2index,
                                        SFO_obj.grid,
                                        grid_size,
                                        SFO_obj.OD2grid,
                                        NameCity,
                                        ODfmaFile,
                                        start,
                                        end,
                                        SFO_obj.save_dir_local,
                                        seconds_in_minute = 60,
                                        )
                    FirstFile = False
                else:
                    df2,_,ROutput = OD_from_fma(SFO_obj.polygon2OD,
                                        SFO_obj.osmid2index,
                                        SFO_obj.grid,
                                        grid_size,
                                        SFO_obj.OD2grid,
                                        NameCity,
                                        ODfmaFile,
                                        start,
                                        end,
                                        SFO_obj.save_dir_local,
                                        seconds_in_minute = 60,
                                        )
                    df1 = pd.concat([df1,df2],ignore_index = True)
    return df1


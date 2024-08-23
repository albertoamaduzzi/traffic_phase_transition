def InitWholeProcessStateFunctions():
    return {"GeometricalSettingsSpatialPartition":False,
        "GetGrid":False,
        "GetBoundariesInterior":False,
        "GetDirectionMatrix":False,
        "GetODGrid":False,
        "GetLattice":False,
        "GetVectorField":False,
        "GetPotentialLattice":False,
        "GetPotentialDataframe":False,
        "SmoothPotential":False,
        "ConvertLattice2PotentialDataframe":False,
        "CompletePotentialDataFrame":False,
        "SavePotentialDataframe":False,
        "SaveVectorField":False,
        "VespignaniBlock":False,
        "ModifyMorphologyCity":False}                

def InitVariablesChecks():
    return False

def ChangeKey2Done(state, key):
    state[key] = True
    return state


from OutputStats_Functions import FillInfoTraffic, Modify_InfoTraffic_Tau_IfAlreadyJammed
def pipeline_fill_InfoTraffic(ErrorExp,                             # NOTE: ErrorExp is the error of the exponential fit for the nt given (City,UCI,R)
                                ErrorPL,                            # NOTE: ErrorPL is the error of the power law fit for the nt given (City,UCI,R)
                                A0,                                 # NOTE: A0 is the amplitude of the exponential fit for the nt given (City,UCI,R)      
                                A_pl,                               # NOTE: A_pl is the amplitude of the power law fit for the nt given (City,UCI,R)
                                alpha_exp,                          # NOTE: alpha_exp is the exponent of the exponential fit for the nt given (City,UCI,R)
                                alpha_pl,                           # NOTE: alpha_pl is the exponent of the power law fit for the nt given (City,UCI,R)
                                City,                               # NOTE: City is the name of the city
                                Gamma,                              # NOTE: Gamma is the value of the gamma computed for the nt given (City,UCI,R)
                                DfUnload,                           # NOTE: DfUnload is the value of the DfUnload computed for the nt given (City,UCI,R)
                                found_jam = False,                  # NOTE: found_jam is a boolean that tells if the jam is already found
                                R = None,
                                UCI = None
        
    ):

    InfoTraffic = FillInfoTraffic(ErrorExp,
                                    ErrorPL,
                                    A0,
                                    A_pl,
                                    alpha_exp,
                                    alpha_pl,
                                    City)
    print(f"{City} Computing Gamma R: {R}, UCI: {UCI}")   
    InfoTraffic["Gamma"] = Gamma
    if not found_jam and InfoTraffic["IsJam"]:
        # Change the value of found_jam to True -> otherwise if already jammed do not touch anything
        found_jam = True    
    InfoTraffic,found_jam = Modify_InfoTraffic_Tau_IfAlreadyJammed(InfoTraffic,found_jam, DfUnload)
    return InfoTraffic, found_jam



try:
    from mpi4py import MPI
    Bmpi = True
    print("mpi available")
except:
    Bmpi = False 
    print("mpi not available")
import concurrent.futures


def MPI_main(NameCities,TRAFFIC_DIR):
    """
        NOTE: Does not work currently.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    FreeFromProcesses = np.array([True] if rank == 0 else np.empty(1,dtype='b'))
    win = MPI.Win.Create(FreeFromProcesses,comm = comm)
    CityName = NameCities[rank]
    City2RUCI = {CityName:{"UCI":[],"R":[]}}
    # Everything is handled inside the object
    GeoInfo = GeometricalSettingsSpatialPartition(NameCities[rank],TRAFFIC_DIR)
    # Compute the Potential and Vector field for non modified fluxes
    UCI = GeoInfo.RoutineVectorFieldAndPotential()
    # Compute the Fit for the gravity model
    GeoInfo.ComputeFit()
    # Initialize the Concatenated Df for Simulation [It is common for all different R]
    GeoInfo.InitializeDf4Sim()
    GeoInfo.ComputeEndFileInputSimulation()
    # NOTE: Can Parallelize this part and launch the simulations in parallel.
    City2RUCI[CityName]["R"] = list(GeoInfo.ArrayRs)
    for R in GeoInfo.ArrayRs:
        # Simulation for the monocentric case.
        NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI,R)
        City2RUCI[CityName]["UCI"].append(UCI)
        while(True):
            # Start a lock into the window (ensures that each process have syncronous access to the window from a queue)
            win.Lock(rank = 0, lock_type = MPI_LOCK_EXCLUSIVE)
            # Tells that the process is occupied
            LaunchDockerFromServer(container_name,CityName,GeoInfo.start,GeoInfo.start + 1,R,UCI)
            DeleteInputSimulation(NotModifiedInputFile)
            win.unlock(rank=0)
            break
        # Generate modified Fluxes
    for cov in GeoInfo.config['covariances']:
        for distribution in ['exponential']:
            for num_peaks in GeoInfo.config['list_peaks']:
                for R in GeoInfo.ArrayRs:        
                    Modified_Fluxes,UCI1 = GeoInfo.ChangeMorpholgy(cov,distribution,num_peaks)
                    City2RUCI[CityName]["UCI"].append(UCI1)
                    ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1,R,Modified_Fluxes)
                    start = GeoInfo.start
                    end = start + 1
                    InputSimulation = (container_name,CityName,start,end,R,UCI1)
                    # Launch the processes one after the other since you may have race conditions on GPUs (we have just two of them)
                    while(True):
                        # Start a lock into the window (ensures that each process have syncronous access to the window from a queue)
                        win.Lock(rank = 0, lock_type = MPI_LOCK_EXCLUSIVE)
                        # Tells that the process is occupied
                        if os.path.isfile(os.path.join(OD_dir,f"{CityName}_oddemand_{start}_{end}_R_{R}_UCI_{round(UCI1,3)}.csv")):
                            logger.info(f"Launching docker, {CityName}, R: {R}, UCI: {round(UCI1,3)}")
                            LaunchDockerFromServer(InputSimulation)
                            DeleteInputSimulation(ModifiedInputFile)
                        win.unlock(rank=0)
                        break
                    # Post Process
                    City2Config = InitConfigPolycentrismAnalysis([CityName])                        
                    PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName])  
                    PCTA.CompleteAnalysis()
                    with open(os.path.join(BaseConfig,'post_processing_' + CityName +'.json'),'w') as f:
                        json.dump(City2Config,f,indent=4)
    win.Free()


def Threaded_main_Futures(rank, num_threads, lock):
    CityName = NameCities[rank]
    City2RUCI = {CityName: {"UCI": [], "R": []}}
    # Everything is handled inside the object
    GeoInfo = GeometricalSettingsSpatialPartition(NameCities[rank], TRAFFIC_DIR)
    GeoInfo.GetGeometries()
    # Compute the Potential and Vector field for non modified fluxes
    UCI = GeoInfo.RoutineVectorFieldAndPotential()
    # Compute the Fit for the gravity model
    GeoInfo.ComputeFit()
    # Initialize the Concatenated Df for Simulation [It is common for all different R]
    GeoInfo.InitializeDf4Sim()
    GeoInfo.ComputeEndFileInputSimulation()
    # NOTE: Can Parallelize this part and launch the simulations in parallel.
    City2RUCI[CityName]["R"] = list(GeoInfo.ArrayRs)

    def process_R(R):
        NotModifiedInputFile = GeoInfo.ComputeDf4SimNotChangedMorphology(UCI, R)
        City2RUCI[CityName]["UCI"].append(UCI)

        with lock:
            LaunchDockerFromServer(container_name, CityName, start, end, R, UCI, verbose)
            DeleteInputSimulation(NotModifiedInputFile)

        def process_cov_distribution_peaks(cov, distribution, num_peaks):
            Modified_Fluxes, UCI1 = GeoInfo.ChangeMorpholgy(cov, distribution, num_peaks)
            City2RUCI[CityName]["UCI"].append(UCI1)
            ModifiedInputFile = GeoInfo.ComputeDf4SimChangedMorphology(UCI1, R, Modified_Fluxes)
            start = GeoInfo.start
            end = start + 1
            InputSimulation = (container_name, CityName, start, end, R, UCI1, verbose)

            with lock:
                if os.path.isfile(os.path.join(OD_dir, f"{CityName}_oddemand_{start}_{end}_R_{R}_UCI_{round(UCI1, 3)}.csv")):
                    logger.info(f"Launching docker, {CityName}, R: {R}, UCI: {round(UCI1, 3)}")
                    LaunchDockerFromServer(InputSimulation)
                    DeleteInputSimulation(ModifiedInputFile)

            # Post Process
            City2Config = InitConfigPolycentrismAnalysis([CityName])
            PCTA = Polycentrism2TrafficAnalyzer(City2Config[CityName])
            PCTA.CompleteAnalysis()
            with open(os.path.join(BaseConfig, 'post_processing_' + CityName + '.json'), 'w') as f:
                json.dump(City2Config, f, indent=4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for cov in GeoInfo.config['covariances']:
                for distribution in ['exponential']:
                    for num_peaks in GeoInfo.config['list_peaks']:
                        futures.append(executor.submit(process_cov_distribution_peaks, cov, distribution, num_peaks))
            concurrent.futures.wait(futures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_R, R) for R in GeoInfo.ArrayRs]
        concurrent.futures.wait(futures)


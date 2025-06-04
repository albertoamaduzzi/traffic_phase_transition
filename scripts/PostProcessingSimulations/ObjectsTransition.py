from DateTimeHandler import *
class TimeInfo:
    """
    Class to store time information for a simulation.
    """

    def __init__(self,
                 int_start_time_simulation_hours: int,
                 int_end_time_simulation_hours: int,
                 int_start_control_group_hours: int,
                 int_end_control_group_hours: int,
                 int_t0_hours: int,
                 delta_t_minutes: int = 10,
                 ):
        """
        Initialize the TimeInfo object.

        :param start_time: Start time of the simulation.
        :param end_time: End time of the simulation.
        """
        # start end simulation
        self.int_start_time_simulation_hours = int_start_time_simulation_hours
        self.int_end_time_simulation_hours = int_end_time_simulation_hours
        self.int_start_time_simulation_minutes = int_start_time_simulation_hours * MINUTES_IN_HOUR
        self.int_end_time_simulation_minutes = int_end_time_simulation_hours * MINUTES_IN_HOUR
        self.int_start_time_simulation_seconds = int_start_time_simulation_hours * SECONDS_IN_MINUTE* MINUTES_IN_HOUR
        self.int_end_time_simulation_seconds = int_end_time_simulation_hours * SECONDS_IN_MINUTE* MINUTES_IN_HOUR
        # start end control group
        self.int_start_control_group_hours = int_start_control_group_hours
        self.int_end_control_group_hours = int_end_control_group_hours
        self.int_start_control_group_minutes = int_start_control_group_hours * MINUTES_IN_HOUR
        self.int_end_control_group_minutes = int_end_control_group_hours * MINUTES_IN_HOUR
        self.int_start_control_group_seconds = int_start_control_group_hours * SECONDS_IN_MINUTE* MINUTES_IN_HOUR
        self.int_end_control_group_seconds = int_end_control_group_hours * SECONDS_IN_MINUTE* MINUTES_IN_HOUR
        # t0 -> NOTE: time when the unload starts
        self.int_t0_hours = int_t0_hours
        self.int_t0_minutes = int_t0_hours * MINUTES_IN_HOUR
        self.int_t0_seconds = int_t0_hours * SECONDS_IN_MINUTE* MINUTES_IN_HOUR
        # delta_t -> NOTE: Interval aggragation
        self.delta_t_minutes = delta_t_minutes
    def __str__(self):
        return f"Start Time: {self.start_time}, End Time: {self.end_time}"
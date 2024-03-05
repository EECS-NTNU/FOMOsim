from policies.fosen_haldorsen.Subproblem.parameters_subproblem import ParameterSub
from policies.fosen_haldorsen.Subproblem.subproblem_model import run_model
import settings
import sim


class ModelManager:

    time_horizon = 25  #this is being overwritten, so not necessarily hardcoded

    def __init__(self, state, vehicle, time_horizon):
        self.state = state
        self.vehicle = vehicle
        self.scores = list()
        self.time_horizon
        ModelManager.time_horizon = time_horizon
        
    def run_one_subproblem(self, route, route_full_set_index, pattern, customer_scenario, weights):
        customer_arrivals = ModelManager.arrivals_after_visit(route, route_full_set_index, customer_scenario)
        L_CS = list()
        L_FS = list()
        base_viol = list()
        base_dev = list()
        V_0 = 0
        D_0 = 0
        for i in range(len(route.stations)):
            st_L_CS, st_L_FS = ModelManager.get_base_inventory(route.stations[i], route.station_visits[i],
                                                               customer_scenario[route_full_set_index[i]])
            if i == 0:
                V_0, D_0 = ModelManager.get_base_violations(self.state, route.stations[i], st_L_CS, st_L_FS, customer_arrivals[i], pattern=pattern)
            st_viol, st_dev = ModelManager.get_base_violations(self.state, route.stations[i], st_L_CS, st_L_FS, customer_arrivals[i])
            L_CS.append(st_L_CS)
            L_FS.append(st_L_FS)
            base_viol.append(st_viol)
            base_dev.append(st_dev)
        params = ParameterSub(self.state, route, self.vehicle, pattern, customer_arrivals, L_CS, L_FS, base_viol, V_0, D_0, base_dev,
                              weights)
        return run_model(params)

    """
    Returns number of customer arrivals for incoming charged bikes, incoming flat bikes, outgoing charged bikes 
    from time of visit to time horizon based on scenario
    """
    @staticmethod
    def arrivals_after_visit(route, route_ind, customer_arrivals):
        arrivals = list()
        for i in range(len(route_ind)):
            arrival_time = route.station_visits[i]
            if arrival_time > ModelManager.time_horizon:
                arrivals.append([0, 0, 0])
            else:
                time_station_arrivals = customer_arrivals[route_ind[i]]
                st_arrivals = list()
                for j in range(len(time_station_arrivals)):
                    split_index = len(time_station_arrivals) - 1
                    for k in range(len(time_station_arrivals[j])):
                        if time_station_arrivals[j][k] > arrival_time:
                            split_index = k
                            break
                    counted_arrivals = time_station_arrivals[j][split_index:]
                    st_arrivals.append(len(counted_arrivals))
                arrivals.append(st_arrivals)
        return arrivals

    """
    Calculate inventory levels at station at time of visit for given customer arrival scenario
    """
    @staticmethod
    def get_base_inventory(station, visit_time_float, customer_arrivals=None):
        # convert from new sim
        station_current_charged_bikes = len(station.get_available_bikes())
        station_current_flat_bikes = len(station.get_swappable_bikes(settings.BATTERY_LIMIT))

        visit_time = int(visit_time_float)
        L_CS = station_current_charged_bikes
        L_FS = station_current_flat_bikes
        incoming_charged_bike_times = customer_arrivals[0]
        incoming_flat_bike_times = customer_arrivals[1]
        outgoing_charged_bike_times = customer_arrivals[2]
        for i in range(visit_time):
            c1 = incoming_charged_bike_times.count(i)
            c2 = incoming_flat_bike_times.count(i)
            c3 = outgoing_charged_bike_times.count(i)
            L_CS = max(0, min(station.capacity - L_FS, L_CS + c1 - c3))
            L_FS = min(station.capacity - L_CS, L_FS + c2)
        return L_CS, L_FS

    """
    Returning base violations from time of visit to time horizon. Assuming optimal sequencing of customer arrivals
    """
    @staticmethod
    def get_base_violations(state, station, visit_inventory_charged, visit_inventory_flat, customer_arrivals, pattern=None):
        incoming_charged_bikes = customer_arrivals[0]
        incoming_flat_bikes = customer_arrivals[1]
        outgoing_charged_bikes = customer_arrivals[2]
        if isinstance(station, sim.Depot):
            return 0, 0
        if pattern:
            starvation = abs(min(0, visit_inventory_charged + pattern[0] - pattern[1] + pattern[3]
                                 + incoming_charged_bikes - outgoing_charged_bikes))
            congestion = max(0, visit_inventory_charged + visit_inventory_flat + incoming_charged_bikes
                             + incoming_flat_bikes - pattern[1] + pattern[3] - pattern[2] + pattern[4]
                             - min(visit_inventory_charged + incoming_charged_bikes, outgoing_charged_bikes)
                             - station.capacity)
            dev = abs(visit_inventory_charged + pattern[0] - pattern[1] + pattern[3] + incoming_charged_bikes
                      - outgoing_charged_bikes + starvation - congestion - station.get_target_state(state.day(), state.hour()))
        else:
            starvation = abs(min(0, visit_inventory_charged + incoming_charged_bikes - outgoing_charged_bikes))
            congestion = max(0, visit_inventory_charged + visit_inventory_flat + incoming_charged_bikes
                             + incoming_flat_bikes - min(visit_inventory_charged + incoming_charged_bikes,
                                                         outgoing_charged_bikes) - station.capacity)
            dev = abs(visit_inventory_charged + incoming_charged_bikes
                      - outgoing_charged_bikes + starvation - congestion - station.get_target_state(state.day(), state.hour()))
        return starvation + congestion, dev

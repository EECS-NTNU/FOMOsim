'''import sim
from typing import List, Tuple, Union
from policies.sjovik_sund.mdp.mdp_formulation import MdpAction
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action

def _generate_operational_profiles(state, vehicle, maintenance_enabled: bool, verbose: bool):
    if verbose:
        print("HEI jeg genererer operasjonelle profiler")
    profiles = []
   
    if vehicle.is_at_depot():
        n_vehicle = len(vehicle.get_bike_inventory())
        vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
        free_cap = max(0, vehicle_capacity - n_vehicle)
        repaired_available = len(getattr(vehicle.location, "fixed_queue", {}))
       
        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'load_from_queue': min(repaired_available, free_cap)
        })
        profiles.append({
            'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0,
            'load_from_queue': 0
        })
        return profiles
       
    functional_bikes = [b for b in vehicle.location.get_bikes() if getattr(b, 'is_available', True)]
    target = round(vehicle.location.get_target_state(state.day(), state.hour()))
   
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
    free_cap = max(0, vehicle_capacity - n_vehicle)
   
    if maintenance_enabled:
        repairable_bikes = vehicle.location.get_unusable_bikes()
        num_repairable = len(repairable_bikes)
    else:
        num_repairable = 0
 
    delta = len(functional_bikes) - target
    rebalancing_qty = 0
    if delta > 0: 
        rebalancing_qty = -min(delta, free_cap)
    elif delta < 0: 
        rebalancing_qty = min(-delta, n_vehicle_func)
 
    if not maintenance_enabled:
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        profiles.append({'rebalancing': int(rebalancing_qty / 2), 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        profiles.append({'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
    else:
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
        profiles.append({'rebalancing': rebalancing_qty, 'onsite_repairs': num_repairable, 'depot_removals': 0, 'load_from_queue': 0})
       
        depot_removals = min(num_repairable, free_cap)
        reb_with_broken = rebalancing_qty
        if delta > 0:
            reb_with_broken = min(delta, max(0, free_cap - depot_removals))
           
        profiles.append({'rebalancing': reb_with_broken, 'onsite_repairs': 0, 'depot_removals': depot_removals, 'load_from_queue': 0})
        profiles.append({'rebalancing': 0, 'onsite_repairs': 0, 'depot_removals': 0, 'load_from_queue': 0})
       
    unique_profiles = []
    seen = set()
    for p in profiles:
        tup = (p['rebalancing'], p['onsite_repairs'], p['depot_removals'], p['load_from_queue'])
        if tup not in seen:
            seen.add(tup)
            unique_profiles.append(p)
           
    if verbose:
        print(f"1. Operational profiles generated: {unique_profiles}")
    return unique_profiles
 
def _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled: bool, n_candidates: int, wide_search: bool, verbose: bool):
    if verbose:
        print("HEI jeg genererer neste rute")
    candidates = []
   
    all_stations = [s for s in state.get_stations() if s.id != vehicle.location.id and s.id not in tabu_list]
   
    _inv = vehicle.get_bike_inventory()
    _n_func_avail = sum(1 for b in _inv if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    _cap = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", max(1, len(_inv)))))
    _cap = max(1, _cap) 
    _free_space = max(0, _cap - len(_inv))
    free_ratio = _free_space / _cap
   
    def score_station(s):
        functional = len([b for b in s.get_bikes() if getattr(b, 'is_available', True)])
        target = round(s.get_target_state(state.day(), state.hour()))
        delta = target - functional 
       
        can_deliver = (delta > 0 and _n_func_avail > 0)
        can_pickup = (delta < 0 and _free_space > 0)
       
        score = 0
        if free_ratio <= 0.2 and can_deliver:
            score = delta 
        elif free_ratio >= 0.8 and can_pickup:
            score = abs(delta) 
        elif 0.2 < free_ratio < 0.8 and (can_deliver or can_pickup):
            score = abs(delta) 
       
        if maintenance_enabled:
            broken = len(s.get_unusable_bikes())
            score += broken * 1.5 
           
        if score > 0:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, s.id)
            if travel_time > 0:
                score = score / (travel_time ** 0.5)
           
        return score
   
    all_stations.sort(key=score_station, reverse=True)

    if wide_search:
        # NN MODE: Diverse candidate pool (Top critical + geographically closest)
        n_top_critical = 8
        n_nearest = 7
        
        top_critical = all_stations[:n_top_critical]
        stations_by_distance = sorted(all_stations, key=lambda s: state.get_vehicle_travel_time(vehicle.location.id, s.id))
        nearest_stations = stations_by_distance[:n_nearest]
        
        # Combine and remove duplicates while preserving order
        combined_pool = top_critical + nearest_stations
        seen = set()
        chosen_stations = []
        for s in combined_pool:
            if s.id not in seen:
                seen.add(s.id)
                chosen_stations.append(s)
        candidates.extend([s.id for s in chosen_stations])
    else:
        # BASELINE MODE: Top critical only
        candidates.extend([s.id for s in all_stations[:n_candidates]])
   
    if maintenance_enabled:
        depots = state.get_depots()
        if depots:
            depots.sort(key=lambda d: state.get_vehicle_travel_time(vehicle.location.id, d.id))
            nearest_depot = depots[0].id
            if nearest_depot not in candidates and nearest_depot not in tabu_list and nearest_depot != vehicle.location.id:
                candidates.append(nearest_depot)
               
    if verbose:
        print(f"2. Routing candidates generated: {candidates}")
    return candidates
 
def generate_candidates(
    state,
    vehicle,
    maintenance_enabled: bool,
    return_pairs: bool = False,
    wide_search: bool = False,
    verbose: bool = False,
    n_candidates: int = 8,
    **kwargs,
) -> Union[List[sim.Action], List[Tuple]]:
    
    if verbose:
        print("HEI jeg sammenslår alt")
 
    tabu_list = [v.location.id for v in state.get_vehicles() if v.id != vehicle.id]
   
    op_profiles = _generate_operational_profiles(state, vehicle, maintenance_enabled, verbose)
    routing_targets = _generate_routing_candidates(state, vehicle, tabu_list, maintenance_enabled, n_candidates, wide_search, verbose)
   
    sim_actions = []
    mdp_actions = []
    cur_id = vehicle.location.id
   
    broken_inventory = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) in ['depot', 'onsite'])
    n_vehicle = len(vehicle.get_bike_inventory())
    n_vehicle_func = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
    vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle)))
   
    if verbose:
        print(f"Vehicle capacity: {vehicle_capacity}, Vehcile inventory: {n_vehicle} (Functional: {n_vehicle_func}, Broken: {broken_inventory})")
        for route_id in routing_targets:
            station = next((s for s in state.get_stations() if s.id == route_id), None)
            if station:
                func_bikes = len([b for b in station.get_bikes() if getattr(b, 'is_available', True)])
                t_state = station.get_target_state(state.day(), state.hour())
                print(f"  Candidate {route_id}: target state={t_state:.2f}, current functional inventory={func_bikes}")
            else:
                print(f"  Candidate {route_id}: (Depot)")
 
    for op in op_profiles:
        broken_after_op = broken_inventory + op['depot_removals']
        functional_after_op = n_vehicle_func - op['rebalancing'] + op['load_from_queue']
        total_after_op = broken_after_op + functional_after_op

        for route in routing_targets:
            is_depot = any(d.id == route for d in state.get_depots())

            if is_depot and broken_after_op == 0 and op['load_from_queue'] == 0:
                continue

            if broken_after_op >= vehicle_capacity and not is_depot:
                continue

            if total_after_op > vehicle_capacity:
                continue
               
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(op['rebalancing']),
                onsite_repairs=int(op['onsite_repairs']),
                depot_removals=int(op['depot_removals']),
                load_from_queue=int(op['load_from_queue']),
                next_station=route
            )
            sim_action = mdp_action_to_sim_action(mdp_action, state, vehicle)
            sim_actions.append(sim_action)
            mdp_actions.append(mdp_action)

    if not sim_actions and routing_targets:
        fallback_mdp = MdpAction(
            current_station=cur_id,
            rebalancing=0, onsite_repairs=0, depot_removals=0, load_from_queue=0,
            next_station=routing_targets[0]
        )
        sim_actions.append(mdp_action_to_sim_action(fallback_mdp, state, vehicle))
        mdp_actions.append(fallback_mdp)

    if verbose:
        print(f"3. Total options generated: {len(sim_actions)}")
    return list(zip(mdp_actions, sim_actions)) if return_pairs else sim_actions'''
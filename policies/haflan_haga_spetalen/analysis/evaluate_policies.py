#!/bin/python3

"""
MAIN FILE FOR RUNNING EVALUATIONS OF WORLD OBJECTS/MODELS

Run from terminal: python analysis/evaluate_policies.py <world_attribute> <directory with models>
make sure it is ran from repository root
"""

import copy
import os
import math
from multiprocessing import Pool
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))

import sim
import policies.haflan_haga_spetalen.value_functions
import policies.haflan_haga_spetalen.analysis.export_metrics_to_xlsx
from visualization.visualizer import visualize_analysis


def run_analysis_from_path(
    path: str,
    visualize_route=False,
    runs_per_policy=10,
    shift_duration=960,
    world_attribute="shift_duration",
    export_to_excel=False,
    multiprocess=True,
):
    """
    Run analysis from the input path
    :param path: string path of folder with models to analyse
    :param visualize_route: is route should vi visualized
    :param runs_per_policy: how many runs to do for every model
    :param shift_duration: length of shift
    :param world_attribute: Attribute to label models in end chart
    :param export_to_excel: if excel export
    :param multiprocess: true if method use multiprocessing
    """
    # Sort the policies by the training duration
    world_objects = sorted(
        [
            sim.Simulator.load(os.path.join(path, world_obj_path))
            for world_obj_path in os.listdir(path)
            if world_obj_path != ".DS_Store"
        ],
        key=lambda world_object: world_object.policy.value_function.shifts_trained
        if hasattr(world_object.policy, "value_function")
        else math.inf,
    )

    for world in world_objects:
        # route visualization
        if visualize_route:
            for event in world.event_queue:
                if isinstance(event, sim.VehicleArrival):
                    event.visualize = True

        # Always rollout for 8 hours
        world.shift_duration = shift_duration

        # Do not learn anything more when during evaluation
        world.disable_training = True

        world.policy.epsilon = 0

        for vehicle in world.state.vehicles:
            vehicle.service_route = []

    return run_analysis(
        world_objects,
        runs_per_policy=runs_per_policy,
        title=path.split("/")[-1],  # Use "last" folder as title in the plot
        world_attribute=world_attribute,
        export_to_excel=export_to_excel,
        multiprocess=multiprocess,
    )


def evaluate_world(world, world_attribute, verbose, runs_per_policy):
    """
    Base method for evaluation of a world object
    :param world: world to analyze
    :param world_attribute: Attribute to label models in end chart
    :param verbose: show console logs
    :param runs_per_policy: how many runs to do for every model
    :return: tuple of world after evaluation and td errors
    """
    # Create label for each model/world
    world.label = (
        f"{world.policy} - {world_attribute}: {getattr(world, world_attribute)}"
    )
    if verbose:
        print(f"\n---------- {world.label} ----------")
    metrics = []
    # Run world run_per_policy times
    for _ in range(runs_per_policy):
        run_world = copy.deepcopy(world)
        # run the world and add the world object to a list containing all world instances
        run_world.run()
        metrics.append(run_world.metrics)
    # Aggregate results from all runs
    world.metrics = metrics[0] # FIXME: should be average of all runs

    # If the model has a value function add the td errors to the output
    td_error_tuple = None
    if hasattr(world.policy, "value_function"):
        td_error_tuple = (
            world.policy.value_function.td_errors,
            world.policy.__str__(),
        )
    return world, td_error_tuple


def run_analysis(
    worlds,
    runs_per_policy=4,
    verbose=True,
    save=False,
    baseline_policy_world=None,
    title=None,
    world_attribute="shift_duration",
    export_to_excel=False,
    multiprocess=True,
):
    """
    Run analysis based on list of world objects
    :param worlds: list of world objects
    :param runs_per_policy: how many runs to do for every model
    :param verbose: show console logs
    :param save: save the world after analysis
    :param baseline_policy_world: world to use for secondary polices
    :param title: Title of graph
    :param world_attribute: Attribute to label models in end chart
    :param export_to_excel: if excel export
    :param multiprocess: true if method use multiprocessing
    """
    instances = []
    if baseline_policy_world:
        first_world = baseline_policy_world
    else:
        first_world, *rest = worlds
    # Always add a policy that does nothing and a random action
    for baseline_policy_class in [
        policies.DoNothing,
        policies.SwapAllPolicy,
        policies.RebalancingPolicy,
    ]:
        # Use the first world as the world for baseline policies
        baseline_policy_world = copy.deepcopy(first_world)
        baseline_policy_world.policy = baseline_policy_class()
        baseline_policy_world.policy.setup_from_state(baseline_policy_world.state)

    worlds.append(baseline_policy_world)

    td_errors_and_label = []
    if multiprocess:
        with Pool() as p:
            results = p.starmap(
                evaluate_world,
                [
                    (world, world_attribute, verbose, runs_per_policy)
                    for world in worlds
                ],
            )
            for world_result, td_error_tuple_result in results:
                td_errors_and_label.append(td_error_tuple_result)
                instances.append(world_result)
    else:
        for world in worlds:
            world_result, td_error_tuple_result = evaluate_world(
                world, world_attribute, verbose, runs_per_policy
            )
            td_errors_and_label.append(td_error_tuple_result)
            instances.append(world_result)

    visualize_analysis(instances, title=title)
    if save:
        for world_result in instances:
            world_result.save_world()

    if export_to_excel:
        policies.haflan_haga_spetalen.analysis.export_metrics_to_xlsx.metrics_to_xlsx(instances)

    return instances


if __name__ == "__main__":
    import sys

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[2]}")
        run_analysis_from_path(
            sys.argv[2],
            world_attribute=sys.argv[1],
            runs_per_policy=1,
            multiprocess=False,
        )
    else:
        run_analysis_from_path(
            "sim_cache/trained_models/ANNValueFunction/c20_s2500/2022-03-04T11:50",
            shift_duration=960,
            runs_per_policy=10,
            multiprocess=False,
        )

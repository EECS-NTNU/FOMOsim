from sim import Simulator
import matplotlib.pyplot as plt
import copy
import sim
import clustering.scripts
import policies
import pandas 

y_results = []

def collect_data(world_instance):
    metrics = world_instance.metrics.get_all_metrics()
    if len(metrics["total_available_scooters"]) > 0:
        x = [item[0] for item in metrics["total_available_scooters"]]
        y = [item[1] for item in metrics["total_available_scooters"]]

    # bare foreløpig her, flyttes ut  
    y_all = pandas.Series(y)
    y_results.append(y_all.mean())
    format_float = "{:.2f}".format(y_all.mean())
    print("mean value:", format_float )
    y_rolling = y_all.rolling(window=1000).mean()  
    plt.plot(x,y_rolling)


def sense_analyze(noSampleRuns):
    """
    Ad hoc sensitivity analysis with visualisation
    """
    for i in range(noSampleRuns):
        print("FOMO simulation-run", i)
        # Set up initial state
        # This is done with a script that reads data from an "entur" snapshot
        state = clustering.scripts.get_initial_state("Bike",
            500,
            5,
            number_of_vans=1,
            number_of_bikes=0,
        )
        # Set up first simulator
        simulator = sim.Simulator(
            10080,
            policies.RebalancingPolicy(),
            copy.deepcopy(state),
            verbose=True,
            visualize=False,
            label="Rebalancing",
        )
        simulator.run()
        collect_data(simulator)
############# "main program" in sense ################################        
    print("Available scooters plot:")
    plt.show()
    print("All results:", y_results)
    y_series = pandas.Series(y_results)
    print("min:", y_series.min(), "max: ", y_series.max(), "mean:", y_series.mean(), "stdev:", y_series.std())
    print("... bye for now")


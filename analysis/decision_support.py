import os

import clustering.helpers
import clustering.methods
import clustering.scripts

from globals import TEST_DATA_DIRECTORY
import numpy as np
import pandas as pd


def battery_analysis():
    """
    Find average used battery for trip
    """
    previous_snapshot = None
    averages = []
    for index, file_path in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        current_snapshot = clustering.methods.read_bounded_csv_file(
            f"{TEST_DATA_DIRECTORY}/{file_path}"
        )
        if previous_snapshot is not None:
            # Join tables on scooter id
            merged_tables = pd.merge(
                left=current_snapshot,
                right=previous_snapshot,
                left_on="id",
                right_on="id",
                how="inner",
            )

            # Filtering out scooters that has moved during the 20 minutes
            moved_scooters = clustering.helpers.get_moved_scooters(merged_tables)
            averages.append(
                np.average(
                    [
                        row["battery_x"] - row["battery_y"]
                        for index, row in moved_scooters.iterrows()
                    ]
                )
            )
        previous_snapshot = current_snapshot
    np.average(averages)
    print(averages)
    print(np.average(averages))


def number_of_clusters_analysis():
    """
    Used to test how many clusters it makes sense to use
    :return:
    """
    state = clustering.scripts.get_initial_state(2500, 50)
    state.visualize_clustering()


if __name__ == "__main__":
    number_of_clusters_analysis()

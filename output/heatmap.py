import numpy as np
import matplotlib.pyplot as plt
import sim

def visualize_heatmap(simulators, metric):
    xx = []
    yy = []
    cc = []

    metrics_list = []

    for location_id in simulators[0].state.get_location_ids():
        metrics_list.append(sim.Metric.merge_metrics([sim.state.locations[location_id].metrics for sim in simulators]))

    maxValue = 0
    for metrics in metrics_list:
        if metrics.get_aggregate_value(metric) > maxValue:
            maxValue = metrics.get_aggregate_value(metric)

    if maxValue == 0:
        print("*** Too short simulation, no trips done, will return without producng a heatmap") # Happens for EH_W22 1 hour simulation
        return

    for location in simulators[0].state.get_locations():
        xx.append(location.get_lon())
        yy.append(location.get_lat())
        color = metrics_list[location.location_id].get_aggregate_value(metric) / maxValue
        cc.append(color)

    if simulators[0].state.mapdata is not None:
        filename = simulators[0].state.mapdata[0]
        bBox = simulators[0].state.mapdata[1]

        image = plt.imread(filename)

        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])

        aspect = aspect_geo / aspect_img

        cm = plt.cm.get_cmap('RdYlBu_r')

        fig, ax = plt.subplots()
        im = ax.scatter(xx, yy, c=cc, s=30, edgecolors="black", cmap=cm)
        ax.set_title("Heatmap for metric '" + metric + "'")
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)

        fig.colorbar(im, ax=ax)

        plt.show()

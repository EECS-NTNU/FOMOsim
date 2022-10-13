import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(simulator, metric):
    xx = []
    yy = []
    cc = []

    for station in simulator.state.locations:
        xx.append(station.get_lon())
        yy.append(station.get_lat())
        color = station.metrics.get_aggregate_value(metric) / simulator.metrics.get_aggregate_value(metric)
        cc.append((color,0,1-color))

    if simulator.state.mapdata is not None:
        filename = simulator.state.mapdata[0]
        bBox = simulator.state.mapdata[1]

        image = plt.imread(filename)

        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])

        aspect = aspect_geo / aspect_img

        fig, ax = plt.subplots()
        ax.scatter(xx, yy, c=cc, s=20)
        ax.set_title('Heatmap')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)

        plt.show()

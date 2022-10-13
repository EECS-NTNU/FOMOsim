import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(simulator):
    for station in simulator.state.stations.values():
        print(station.get_lat(), station.get_lon())

    # bBox = ((46.5309, 46.8690, 24.5562, 24.9353))

    # mapImage = plt.imread("map.png")

    # x = [46.659107,
    #      46.702409,
    #      46.712409,
    #      46.722409,
    #      46.732409]

    # y = [24.768269,
    #      24.680454,
    #      24.680454,
    #      24.680454,
    #      24.680454]

    # fig, ax = plt.subplots(figsize = (8,7))
    # ax.scatter(x, y, c='black', s=40)
    # ax.set_title('Plotting Spatial Data on Riyadh Map')
    # ax.set_xlim(bBox[0],bBox[1])
    # ax.set_ylim(bBox[2],bBox[3])
    # ax.imshow(mapImage, zorder=0, extent = bBox, aspect= 'equal')

    # plt.show()

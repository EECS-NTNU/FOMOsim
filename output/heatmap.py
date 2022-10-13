import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(simulator):
    minX = None
    maxX = None
    minY = None
    maxY = None
    xx = []
    yy = []
    cc = []

    for station in simulator.state.locations:
        x = station.get_lon()
        y = station.get_lat()
        if (minX is None) or (x < minX): minX = x
        if (maxX is None) or (x > maxX): maxX = x
        if (minY is None) or (y < minY): minY = y
        if (maxY is None) or (y > maxY): maxY = y
        xx.append(x)
        yy.append(y)
        color = 1
        cc.append((color,0,1-color))

    if simulator.state.mapdata is not None:
        print(simulator.state.mapdata)
        filename = simulator.state.mapdata[0]
        bBox = simulator.state.mapdata[1]

        image = plt.imread(filename)
        aspect = (maxX-minX)/(maxY-minY)
        fig, ax = plt.subplots()
        ax.scatter(xx, yy, c=cc, s=20)
        ax.set_title('Heatmap')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)

        plt.show()

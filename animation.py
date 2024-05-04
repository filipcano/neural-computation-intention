import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MultipleLocator
import json



# Function to update the plot for each frame of the animation
class Animation:
    def __init__(self, params_file):
        # params_file can be a dict or a path to a json file
        if isinstance(params_file, str):
            with open(params_file, 'r') as fp:
                params = json.load(fp)
        else:
            params = params_file
        
        # Define dimensions
        self.road_start = params['min_street_length']
        self.road_end = params['max_street_length']
        self.lower_sidewalk_start = 0
        self.lower_sidewalk_end = params['sidewalk_height']
        self.upper_sidewalk_start = params['sidewalk_height'] + params['crosswalk_width']
        self.upper_sidewalk_end = params['sidewalk_height'] + params['crosswalk_width'] + 1
        self.crosswalk_start = params['crosswalk_pos']
        self.crosswalk_end = params['crosswalk_pos'] + params['crosswalk_width']

    # Function to update the plot for each frame of the animation
    def update(self, frame):
        ax.clear()
        ax.plot([self.road_start, self.road_end], [self.lower_sidewalk_end, self.lower_sidewalk_end], 'k-', lw=2)  # Road
        ax.fill_between([self.road_start, self.road_end], [0, 0], np.array([1,1])*self.lower_sidewalk_end, color='gray', alpha = 0.5)
        ax.fill_between([self.road_start, self.road_end], np.array([1, 1])*self.upper_sidewalk_start, np.array([1,1])*self.upper_sidewalk_end, color='gray', alpha = 0.5)
        # ax.fill_between([road_start, road_end], [0, 0], [1,1]*lower_sidewalk_end )
        ax.plot([self.road_start, self.road_end], [self.upper_sidewalk_start, self.upper_sidewalk_start], 'k-', lw=2)  # Road
        for i in range(self.lower_sidewalk_end, self.upper_sidewalk_start):
            ax.plot([self.crosswalk_start, self.crosswalk_end], [i+0.5, i+0.5], 'r-', lw=2)  # Crosswalk
        

        car_img = plt.imread('images/car.png')
        ped_img = plt.imread('images/ped.png')
        car = OffsetImage(car_img, zoom=1, alpha = 0.5)
        ped = OffsetImage(ped_img, zoom=0.05, alpha = 0.8)
        ax.plot([frame[0]], [5], 'bo', markersize=2)  # Car
        ax.plot([frame[2]], [frame[3]], 'ro', markersize=2)  # Pedestrian
        ax.add_artist(AnnotationBbox(car, [frame[0], 5], zorder=0))
        ax.add_artist(AnnotationBbox(ped, [frame[2], frame[3]], zorder=0))
        
        
        ax.xaxis.set_major_locator(MultipleLocator(1))  # Set major x-axis gridlines every 5 units
        ax.yaxis.set_major_locator(MultipleLocator(1))  # Set major y-axis gridlines every 1 unit
        # ax.xaxis.set_minor_locator(MultipleLocator(1))  # Set minor x-axis gridlines every 1 unit
        # ax.yaxis.set_minor_locator(MultipleLocator(1))  # Set minor y-axis gridlines every 0.2 units
        ax.grid(True, which='major')
        
        ax.set_xlim(0, 70)
        ax.set_ylim(-1, 14)

    def parse_trace_file(self, trace_file):
        df = pd.read_csv(trace_file, sep = ' ')
        trace = []
        for i in df.index:
            trace.append((df.loc[i, 'car_x'] +0.5, df.loc[i,'car_v']+0.5, df.loc[i, 'ped_x']+0.5, df.loc[i, 'ped_y']+0.5))
        return trace

    # Function to create the animation
    def create_animation(self, trace_file):
        trace = self.parse_trace_file(trace_file)
        fig = plt.figure(figsize=(20,5))
        global ax
        ax = fig.add_subplot(111)
        ani = animation.FuncAnimation(fig, self.update, frames=trace, blit=False, repeat=False)
        ani.save('animation.gif', writer='imagemagick', fps=5) # Save as gif
        # plt.show()

    

        




def main():
    # Create the animation
    A = Animation('params_files/params_example.json')
    A.create_animation('trace.txt')



if __name__ == "__main__":
    main()
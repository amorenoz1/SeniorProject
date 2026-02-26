import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

root_file = "RootFiles/output20250506atnov8_dataTree01.root"
output_file = "output3Dplots.pdf"

# Opens the file
try:
    file = uproot.open(root_file)
    tree = file["THit"]
except Exception as e:
    print(f"Error opening ROOT file '{root_file}': {e}")
    exit()


def process_event(tree, event_number, adc_max):

    # Stores the branches in an array
    try:
        data = tree.arrays(
                ["adc0", "adc1", "adc2", "adc3", "adc4", "adc5", "adc6", "adc7", "adc8", "adc9", "adc10", "adc11", 
                 "strip", "detID", "planeID"],
                entry_start=event_number - 1,
                entry_stop=event_number,
                library="ak",
                )
    except Exception as e:
        print(f"Error loading event {event_number}: {e}")
        return

    # Creates a matrix of adc values for 12 time bins
    adc_list = [ak.to_numpy(data[f"adc{i}"][0]) for i in range(12)]
    adc = np.column_stack(adc_list)

    # Strip numebrs for each hit
    strips = ak.to_numpy(data["strip"][0])

    # Create a new figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Gets total hits and time bins
    total_hits = len(strips)
    print(total_hits)

    total_bins = adc.shape[1]

    # Creates the histogram
    for i in range(total_hits):
        for t in range(total_bins):
            adc_val = adc[i, t]
            if adc_val > adc_max:
                ax.bar3d(x=t, y=strips[i], z=0, dx=1, dy= 1, dz=adc_val, shade=True)

    # Set labels
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Strip")
    ax.set_zlabel("ADC")
    ax.set_title("Tracker 0-3")

    # Places the figure in the pdf
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)
        plt.close()

# 1226
process_event(tree, 1226, 0)

'''
# Iterates over the four trackers
for i in range(1):

    # Test data
    time_bins = np.random.randint(0,16, 100)
    strip_numbers = np.random.randint(0, 121, 100)
    adc_values = np.random.randint(0, 5000, 100)

    # Creates a new figure
    tracker_num = i
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create points
    hist, xedges, yedges = np.histogram2d(time_bins, strip_numbers, bins=[np.arange(0, 17), np.arange(0, 62)], weights=adc_values)
    xpos, ypos = np.meshgrid(np.arange(16), np.arange(61), indexing="ij")
    zpos = np.zeros_like(xpos.ravel())

    # Create bars
    ax.bar3d(xpos.ravel(), ypos.ravel(), zpos, 1, 1, hist.ravel(), shade=True)

    # Set labels
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Strip")
    ax.set_zlabel("ADC")
    ax.set_title("Tracker " + str(tracker_num))

    # Define bins for the histogram

    # Places the figure in the pdf
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)
        plt.close()
'''

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

root_file = "output20250506atnov8_dataTree01.root"
output_file = "outputpositions.pdf"

# Opens the root file
try:
    file = uproot.open(root_file)
    tree = file["THit"]
except Exception as e:
    print(f"Error opening ROOT file '{root_file}': {e}")
    exit()


def get_position(adcs, strips, detectors, planes, detector_ID):

    # Filters the hits by detector and separates the planes
    strips_x = []
    strips_y = []
    adc_x = []
    adc_y = []

    for i in range(len(strips)):
        # Filters x hits
        if detectors[i] == detector_ID and planes[i] == 0:
            strips_x.append(strips[i])
            adc_x.append(adcs[i])
        # Filters y hits
        elif detectors[i] == detector_ID:
            strips_y.append(strips[i])
            adc_y.append(adcs[i])

    # Flattens the adc matrices into 1d for taking the overall max
    adc_x = np.concatenate(np.array(adc_x))
    strips_x = np.array(strips_x)
    adc_y = np.concatenate(np.array(adc_y))
    strips_y = np.array(strips_y)

    #Tests
    #print(adc_x)
    #print(strips_x)

    # Temporary method takes max adc to get strip (replace with center of mass later)
    # The //12 determines the strip since the original matrices were flattened
    x_pos = strips_x[np.argmax(adc_x)//12]
    y_pos = strips_y[np.argmax(adc_y)//12]

    return x_pos, y_pos


def create_plot(adcs, strips, detectors, planes, adc_max):

    # Get and plot the hit positions for each detector
    colors = ["red", "blue", "yellow", "green"]

    for i in range(4):
        x_pos, y_pos = get_position(adcs, strips, detectors, planes, i)
        plt.scatter(x_pos, y_pos, color=colors[i], label="Detector " + str(i))

    # Labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Hit Positions for a Single Event")

    # Legend and xy ranges
    plt.legend()
    plt.xlim(0, 128)
    plt.ylim(0, 128)
    plt.grid(True)

    # Saves the plot to a pdf
    pdf.savefig()
    plt.close()


def process_event(event_number, adc_max):

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

    # Inverts the rows and columns so instead of 12 lists of hits, there is an array of size 12 for every hit where the index of the internal array corresponds to the time bin
    all_adc = np.column_stack(adc_list)

    # Strip numbers, detectors, and planes for each hit
    all_strips = ak.to_numpy(data["strip"][0])
    all_detectors = ak.to_numpy(data["detID"][0])
    all_planes = ak.to_numpy(data["planeID"][0])

    # Tests
    #print(all_adc.shape)
    #print(all_strips)
    #print(all_planes.shape)
    #print(all_detectors.shape)

    create_plot(all_adc, all_strips, all_detectors, all_planes, adc_max)


# 1226 is a good event
with PdfPages(output_file) as pdf:
    process_event(1226, 0)

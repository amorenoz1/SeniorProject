import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

root_file = "output20250506atnov8_dataTree01.root"
output_file = "output3Dplots.pdf"

# Opens the root file
try:
        file = uproot.open(root_file)
        tree = file["THit"]
except Exception as e:
        print(f"Error opening ROOT file '{ROOT_FILE_PATH}': {e}")
        exit()


def create_plot(adcs, strips, detectors, planes, detector_ID, plane, adc_max):

        # Filters the adc by detector and plane
        strips_f = []
        adc_f = []
        for i in range(len(strips)):
                if detectors[i] == detector_ID and planes[i] == plane:
                        strips_f.append(strips[i])
                        adc_f.append(adcs[i])
        if len(strips_f) == 0:
                print(f"no hits on this detector and plane")
                return
        strips_f = np.array(strips_f)
        adc_f = np.array(adc_f)

        # Create a new figure
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Gets total hits and time bins
        total_hits = len(strips_f)
        total_bins = adc_f.shape[1]

        # Sets the color to blue for x and red for y
        if plane == 0:
                c = "blue"
        else:
                c = "red"

        # Creates the histogram
        for i in range(total_hits):
                for t in range(total_bins):
                        adc_val = adc_f[i, t]
                        if adc_val > adc_max:
                                ax.bar3d(x=t, y=strips_f[i], z=0, dx=0.9, dy= 0.9, dz=adc_val, color=c, shade=True)

        # Set labels
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Strip")
        ax.set_zlabel("ADC")
        ax.set_title(f"Detector: {detector_ID} Plane: {plane}")

        # Saves the figure to the pdf
        pdf.savefig(fig)
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
        all_adc = np.column_stack(adc_list)

        # Strip numebrs, dectectors, and planes for each hit
        all_strips = ak.to_numpy(data["strip"][0])
        all_detectors = ak.to_numpy(data["detID"][0])
        all_planes = ak.to_numpy(data["planeID"][0])

        create_plot(all_adc, all_strips, all_detectors, all_planes, 0, 0, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 0, 1, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 1, 0, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 1, 1, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 2, 0, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 2, 1, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 3, 0, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 3, 1, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 4, 0, adc_max)
        create_plot(all_adc, all_strips, all_detectors, all_planes, 4, 1, adc_max)


# 1226 is a good event
with PdfPages(output_file) as pdf:
        process_event(1226, 0)

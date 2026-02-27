import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt


N_TIMEBINS = 12
N_STRIPS = 128

def filter_events(data, adc_branches, threshold):
    good_events = []
    for event in range(len(data["evtID"])):
        n = len(data["strip"][event])
        adcs = [data[adc][event] for adc in adc_branches]

        above_threshold = False
        for i in range(n):
            strip_adcs_sum = sum([int(adcs[x][i]) for x in range(12)])
            if strip_adcs_sum >= threshold:
                above_threshold |= True

        if above_threshold:
            good_events.append(event)

    return good_events

def plot_detector_3d_hist(detectors, det_id, plane_id):
    """
    plane_id: 0 = X, 1 = Y
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    dx = 0.7   # timebin width
    dy = 0.7  # strip width

    for strip, plane, adcs in detectors[det_id]:
        if plane != plane_id:
            continue

        for timebin in range(N_TIMEBINS):
            x = timebin
            y = strip
            z = 0                  # bars start at ADC = 0
            dz = adcs[timebin]     # ADC is now the height (Z)

            if dz > 0:
                ax.bar3d(x, y, z, dx, dy, dz)   # ← single uniform color

    # Explicit detector geometry
    ax.set_xlim(0, N_TIMEBINS)
    ax.set_ylim(0, N_STRIPS)

    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Strip")
    ax.set_zlabel("ADC")

    plane_name = "X" if plane_id == 0 else "Y"
    ax.set_title(f"Detector {det_id} – Plane {plane_name} (3D Histogram)")

    plt.tight_layout()
    plt.show()


def beam_reconstruction(root_file_path):
    file = uproot.open(root_file_path)

    tree = file["THit"]

    adc_branches = [f"adc{i}" for i in range(12)]

    branches = ["evtID", "strip", "detID", "planeID"] + adc_branches

    data = tree.arrays(branches, library="np")

    good_events = filter_events(data, adc_branches, threshold=1000 * 12)
    print(good_events)

    for event in good_events:
        strips = data["strip"][event]
        detids = data["detID"][event]
        planes = data["planeID"][event]
        adcs = [data[adc][event] for adc in adc_branches]


        detectors = [[] for _ in range(6)]
        n = len(strips)

        for i in range(n):
            detectors[detids[i]].append((strips[i], planes[i], [int(adcs[x][i]) for x in range(12)]))

        plot_detector_3d_hist(detectors, 0, 0)
        plot_detector_3d_hist(detectors, 0, 1)

        


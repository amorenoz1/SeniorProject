import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
from time import time

from threading import Thread, Semaphore

class Event:
    def __init__(self, ident):
        self.detector_0_x = []
        self.detector_0_y = []
        self.detector_1_x = []
        self.detector_1_y = []
        self.detector_2_x = []
        self.detector_2_y = []
        self.detector_3_x = []
        self.detector_3_y = []
        self.detector_5_u = []
        self.detector_5_v = []
        self.id = ident

    def preprocess(self, strips, adcs, detector_ids, planes, adc_threshold):

        for i in range(len(strips)):
            if np.sum([adcs[j][i] for j in range(len(adcs))]) <= len(adcs) * adc_threshold:
                continue

            data = (strips[i], [adcs[j][i] for j in range(len(adcs))])

            match  detector_ids[i]:
                case 0:
                    if planes[i] == 0:
                        self.detector_0_x.append(data)
                    else:
                        self.detector_0_y.append(data)
                case 1:
                    if planes[i] == 0:
                        self.detector_1_x.append(data)
                    else:
                        self.detector_1_y.append(data)
                case 2:
                    if planes[i] == 0:
                        self.detector_2_x.append(data)
                    else:
                        self.detector_2_y.append(data)
                case 3:
                    if planes[i] == 0:
                        self.detector_3_x.append(data)
                    else:
                        self.detector_3_y.append(data)
                case 4:
                    pass
                case 5:
                    # Maps temporal to spatial strip number
                    temporal = strips[i]
                    if temporal < 128:
                        '''if temporal < 64:
                            mapped = temporal * 2
                        else:
                            mapped = 255 - (temporal * 2)'''
                        data = (temporal//2, [adcs[j][i] for j in range(len(adcs))])

                        # Separate into u and v based on parity
                        if temporal % 2 == 1:
                            self.detector_5_u.append(data)
                        else:
                            self.detector_5_v.append(data)

    def empty(self):
        return (len(self.detector_0_x) + len(self.detector_0_y)
             + len(self.detector_1_x) + len(self.detector_1_y)
             + len(self.detector_2_x) + len(self.detector_2_y)
             + len(self.detector_3_x) + len(self.detector_3_y)
             + len(self.detector_5_u) + len(self.detector_5_v) <= 0)


def preprocessing_batch(data, event_range, good_events, threshold, good_events_mutex,):
    adc_branches = [f"adc{i}" for i in range(15)]

    for event_id in range(event_range[0], event_range[1]):
        e = Event(event_id)

        strips = data["strip"][e.id]
        detids = data["detID"][e.id]
        planes = data["planeID"][e.id]
        adcs = [data[adc][e.id] for adc in adc_branches]

        e.preprocess(strips,adcs,detids,planes,threshold)

        if e.empty():
            continue

        good_events_mutex.acquire()
        good_events.append(e)
        good_events_mutex.release()


def preprocessing(root_file_path, threshold):
    file = uproot.open(root_file_path)
    tree = file["THit"]
    adc_branches = [f"adc{i}" for i in range(15)]
    branches = ["evtID", "strip", "detID", "planeID"] + adc_branches
    data = tree.arrays(branches, library="np")

    good_events = []

    n = len(data["evtID"])
    batch_size = 500

    for k in range(0, n//batch_size, 4):
        mutex = Semaphore()

        threads = []

        for i in range(k, k + 4):
            start = i * batch_size
            end = min(start + batch_size - 1, n)

            print(f"processing event batch: ({start}, {end})")

            t = Thread(target=preprocessing_batch, args=(data, (start, end), good_events, threshold, mutex))
            t.start()
            threads.append(t)

            if end == n:
                break

        for t in threads:
            t.join()

    return good_events


def get_hits(detector_x, detector_y):
    hits_x = [hit[0] for hit in detector_x]
    subset_x = []
    for h in hits_x:
        if h >= 0 and h < 128: subset_x.append(h)
    hits_y = [hit[0] for hit in detector_y]
    subset_y = []
    for h in hits_y:
        if h >= 0 and h < 128: subset_y.append(h)

    return subset_x, subset_y


def hit_hists(root_file_path, threshold, detector):

    grid_x = []
    grid_y = []

    good_events = preprocessing(root_file_path, threshold)

    for event in good_events:
        hits = []
        match detector:
            case 0:
                hits_x, hits_y = get_hits(event.detector_0_x, event.detector_0_y)

            case 1:
                hits_x, hits_y = get_hits(event.detector_1_x, event.detector_1_y)

            case 2:
                hits_x, hits_y = get_hits(event.detector_2_x, event.detector_2_y)

            case 3:
                hits_x, hits_y = get_hits(event.detector_3_x, event.detector_3_y)

            case 4:
                pass

            case 5:
                hits_x, hits_y = get_hits(event.detector_5_u, event.detector_5_v)

        grid_x.extend(hits_x)
        grid_y.extend(hits_y)
    return grid_x, grid_y


def plot_hits(grid_x, grid_y, plot_name):
    fig, axes = plt.subplots(2,1)
    axes[0].hist(grid_x, bins=range(min(grid_x), max(grid_x)+2), rwidth=0.9, color='blue', align='left')
    axes[1].hist(grid_y, bins=range(min(grid_y), max(grid_y)+2), rwidth=0.9, color='green', align='left')
    #axes[0].xticks(range(min(grid_x), max(grid_x)+1))
    axes[0].set_title("Plane 0 hits")
    #axes[1].xticks(range(min(grid_y), max(grid_y)+1))
    axes[1].set_title("Plane 1 hits")
    plt.tight_layout()
    plt.show()



detector = int(sys.argv[1])
threshold = int(sys.argv[2])
root_file = "RootFiles/nov8.root"
'''
        The nov 8 detector 4 data has hits in the range 0-127
        (chips 8). Detector 5 has not been investigated.
'''

#root_file = "output20250506_jan15_flip9flip13_dataTree01.root"
'''
        The jan 15 detector 4 data has hits in the ranges 384-511
        and 896-1024 (chips 11 and 15). Detector 5
'''


start_t = time()
grid_x, grid_y = hit_hists(root_file, threshold, detector)
end_t = time()

print(f"time: {end_t - start_t}")

plot_hits(grid_x, grid_y, "detector: {detector} adc threshold: {threshold}")

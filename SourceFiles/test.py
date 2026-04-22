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
        self.detector_5_
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

    def empty(self):
        return (len(self.detector_0_x) + len(self.detector_0_y)
             + len(self.detector_1_x) + len(self.detector_1_y)
             + len(self.detector_2_x) + len(self.detector_2_y)
             + len(self.detector_3_x) + len(self.detector_3_y)
             <= 0)

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
    hit_ordered_pairs = []

    for hit_x in detector_x:
        max_similarity = -np.inf
        best_pair = None

        for hit_y in detector_y:
            similarity = np.dot(hit_x[1], hit_y[1])
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (hit_x, hit_y)

        hit_ordered_pairs.append(best_pair)

    return hit_ordered_pairs

def get_center_of_mass(adcs):
    weighted_sum = 0
    total_mass = 0

    for l, adc in enumerate(adcs):
        weighted_sum += adc * (l + 1)
        total_mass += adc

    return int(np.round(weighted_sum/total_mass)) - 1

def get_hit_adc(adcs_x, adcs_y):
    x_i = get_center_of_mass(adcs_x)
    y_i = get_center_of_mass(adcs_y)
    return np.max(adcs_x[x_i] + adcs_y[y_i])

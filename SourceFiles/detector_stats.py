
import uproot 
import numpy as np
import matplotlib.pyplot as plt
import sys

LOWER_BOUND_THRESHOLD = 50
UPPER_BOUND_THRESHOLD = 2000

def main():
    if len(sys.argv) != 3:
        print("usage: python3 test.py <root filepath> <detector>")
        return

    detector: int = int(sys.argv[2])
    root_fp: str =  sys.argv[1]
    root_file =  uproot.open(root_fp)  
    root_tree = root_file["THit"]

    adc_branches = [f"adc{i}" for i in range(15)]
    branches = ["evtID", "strip", "detID", "planeID"] + adc_branches
    df = root_tree.arrays(branches, library="np")

    event_amount = len(df["evtID"])

    min_strip = 0
    max_strip = 0
    # max_average_adc = 0
    strip_set_plane_1: set[int] = set()
    strip_set_plane_0: set[int] = set()
    plane_0 = 0
    plane_1 = 0
    for i in range(event_amount):
        adcs =  [df[adc][i] for adc in adc_branches]
        strips = df["strip"][i]
        det_ids = df["detID"][i]
        planes = df["planeID"][i]

        det5_indices = []

        for x in range(len(det_ids)):
            if det_ids[x] == detector:
                det5_indices.append(x)

        # det5_adcs = [adcs[x] for x in det5_indices]
        det5_strips = [strips[x] for x in det5_indices]
        det5_planes = [planes[x] for x in det5_indices]

        for x in det5_indices:
            if planes[x] == 1:
                strip_set_plane_1.add(strips[x])
            else:
                strip_set_plane_0.add(strips[x])


        min_strip = np.min(det5_strips + [min_strip])
        max_strip = np.max(det5_strips + [max_strip])
        for plane in det5_planes: 
            if plane == 0: 
                plane_0 += 1 
            else: 
                plane_1 += 1

    print(f" min strip = {min_strip}\n max strip = {max_strip}\n plane 0 = {plane_0}\n plane 1 = {plane_1}\n unique strips in plane 0 = {len(strip_set_plane_0)}\n unique strips in plane 1 = {len(strip_set_plane_1)}")
    
    l = list(strip_set_plane_1)
    l.sort()
    l1 = [int(x) for x in l]

    print("plane 1 strips")
    print(l1)
    
    l = list(strip_set_plane_0)
    l.sort()
    l1 = [int(x) for x in l]

    print("plane 0")
    print(l1)
        

        



if __name__ == "__main__":
    main()

import uproot 
import heapq
import numpy as np
import matplotlib.pyplot as plt
import sys

LOWER_BOUND_THRESHOLD = 0
APV_OFFSET = 128
GRID = 768
UP_RIGHT = np.array([1, 1])
UP_LEFT = np.array([-1, 1])
DOWN_RIGHT = np.array([1, -1])
DOWN_LEFT = np.array([-1, -1])


even_func_apv8 = lambda n: n // 2
odd_func_apv8 = lambda n: (abs(n - 128) // 2) + 64 
even_func_apv9 = lambda n: n // 2 + 64
odd_func_apv9 = lambda n: (abs(n - 128) // 2)
even_func_apv12 = lambda n: (abs(n  - 126) // 2)
odd_func_apv12 = lambda n: ((n - 1) // 2) + 64
even_func_apv13 = lambda n:(abs(n - 126) // 2) +64
odd_func_apv13 = lambda n: (n - 1) // 2

def get_center_of_mass(adcs):
    weighted_sum = np.int64(0)
    total_mass = np.int64(0)

    for l, adc in enumerate(adcs):
        weighted_sum += np.int64(adc) * l
        total_mass += np.int64(adc)

    idx = int(np.round(weighted_sum / total_mass))
    return max(0, min(len(adcs) - 1, idx))

def get_apv(strip):
    if strip >= 0 and strip < 128: return 0
    if strip >= 128 and strip < 256: return 1
    if strip >= 256 and strip < 384: return 2
    if strip >= 512 and strip < 640: return 4
    if strip >= 640 and strip < 768: return 5
    if strip >= 768 and strip < 896: return 6
    return 0

def get_global_mapped(strip):
    apv = get_apv(strip)
    n = strip - (apv * APV_OFFSET)
    local_map = get_local_mapped(n, apv)
    global_map = local_map + (apv * APV_OFFSET)  if apv < 4 else local_map + ((apv - 1) * APV_OFFSET)
    return global_map

def get_direction(strip, global_map):
    apv = get_apv(strip)
    if apv < 4 and global_map % 2 == 0: return UP_RIGHT
    if apv < 4 and global_map % 2 != 0: return DOWN_RIGHT
    if apv >= 4 and global_map % 2 == 0: return UP_LEFT
    if apv >= 4 and global_map % 2 != 0: return DOWN_LEFT
    return np.array([0 , 0])

def get_x(strip):
    apv = get_apv(strip)
    if apv < 4: return 0
    else: return 383

def get_y(strip, global_map):
    return global_map if get_apv(strip) < 4 else global_map - 384

def get_local_mapped(n, apv):
    match apv:
        case 0:
            if n % 2 == 0:
                return even_func_apv8(n)
            else:
                return odd_func_apv8(n)
        case 1:
            if n % 2 == 0:
                return even_func_apv9(n)
            else:
                return odd_func_apv9(n)
        case 2:
            if n % 2 == 0:
                return even_func_apv8(n)
            else:
                return odd_func_apv8(n)
        case 4:
            if n % 2 == 0:
                return even_func_apv12(n)
            else:
                return odd_func_apv12(n)
        case 5:
            if n % 2 == 0:
                return even_func_apv13(n)
            else:
                return odd_func_apv13(n)
        case 6:
            if n % 2 == 0:
                return even_func_apv12(n)
            else:
                return odd_func_apv12(n)

    return 0


def collect_valid_strips(adcs, strips, det_ids, planes):
    """One pass over the hits of a single event.

    Returns the strips on detector 5 that survive the ADC-sum and plane cuts,
    already decorated with the geometry and peak ADC each strip contributes,
    so nothing has to be recomputed per pair later.
    """
    n_adc = len(adcs)
    lo = LOWER_BOUND_THRESHOLD

    valid = []
    for x in range(len(det_ids)):
        if det_ids[x] != 5:
            continue

        column = [adcs[j][x] for j in range(n_adc)]
        adcs_sum = np.sum(column)
        if adcs_sum < lo:
            continue

        strip = strips[x]
        # apv = get_apv(strip)
        # allowed = [0, 1, 2]
        # if not apv in allowed:
        #     continue
        global_map = get_global_mapped(strip)
        valid.append((
            strip,
            get_x(strip),
            get_y(strip, global_map),
            get_direction(strip, global_map),
            column,
        ))

    return valid


def accumulate_hits(valid, counts, total):
    """Intersect every ordered pair of strips and bin the result straight away."""
    # ordered pairs (i != k) are kept so the counts match the original script;
    # switching to `for k in range(i + 1, len(valid))` gives the same picture
    # at half the weight and half the work.
    heap = []
    counter = 0
    for i in range(len(valid)):
        _, xa, ya, direction_a, peak_a = valid[i]
        for k in range(len(valid)):
            if i == k:
                continue
            _, xb, yb, direction_b, peak_b = valid[k]

            D = np.array([
                [direction_a[0], -direction_b[0]],
                [direction_a[1], -direction_b[1]]
            ])
            p = np.array([xb - xa, yb - ya])

            try:
                s_t = np.linalg.solve(D, p)
            except np.linalg.LinAlgError:
                continue

            l1 = np.array([xa, ya]) + s_t[0] * direction_a
            l2 = np.array([xb, yb]) + s_t[1] * direction_b
            assert np.allclose(l1, l2)

            if l1[0] < 0 or l1[0] >= 384 or l1[1] < 0 or l1[1] >= 384:
                continue

            similarity_score = np.dot(np.linalg.norm(peak_a), np.linalg.norm(peak_b))
            adc = max(peak_a[get_center_of_mass(peak_a)], peak_b[get_center_of_mass(peak_b)])
            heapq.heappush(heap,(-similarity_score, counter, (l1, adc)))
            counter += 1
            

    if (len(heap) <= 0): 
        return
    
    similarity_score, i, best_match = heap[0]
    x, y = int(best_match[0][0] * 2), int(best_match[0][1] * 2)
    counts[y, x] += 1
    total[y, x] += best_match[1]


def main():
    if len(sys.argv) != 2:
        print("usage: python3 test.py <root filepath>")
        return

    root_fp: str =  sys.argv[1]
    root_file =  uproot.open(root_fp)  
    root_tree = root_file["THit"]

    adc_branches = [f"adc{i}" for i in range(15)]
    branches = ["evtID", "strip", "detID", "planeID"] + adc_branches
    df = root_tree.arrays(branches, library="np")

    event_amount = len(df["evtID"])

    counts = np.zeros((GRID, GRID), dtype=np.int64)
    total = np.zeros((GRID, GRID))
    good_event_count = 0

    for i in range(event_amount):
        adcs = [df[adc][i] for adc in adc_branches]
        valid = collect_valid_strips(adcs, df["strip"][i], df["detID"][i], df["planeID"][i])

        if len(valid) <= 1:
            continue

        good_event_count += 1
        accumulate_hits(valid, counts, total)

    print(good_event_count)

    r, c = np.unravel_index(np.argmax(total), total.shape)
    print(f"hottest: y={r} x={c}  n={counts[r,c]}  sum={total[r,c]:.0f}  "
      f"mean={total[r,c]/counts[r,c]:.0f}")

    view = np.where(counts > 0, total, np.nan)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#1a1a1a')

    vmax = np.percentile(total[counts > 0], 98)
    plt.imshow(view, cmap=cmap, vmin=0, vmax=vmax,
               origin='lower', aspect='equal', interpolation='nearest')
    plt.colorbar(label='Accumulated ADC', extend='max')

    plt.show()
if __name__ == "__main__":
    main()


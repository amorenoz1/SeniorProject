import uproot
import numpy as np
import matplotlib.pyplot as plt
from time import time

from threading import Thread, Semaphore

# ---------------------------------------------------------------------------
# Detector 5 remap (from cylindrical_mapping.py)
# ---------------------------------------------------------------------------
OUTER_APVS = {8, 10, 12, 14}
MIDDLE_APVS = {9, 13}

def global_to_apv(global_strip):
    if 0 <= global_strip <= 127:
        return 8
    if 128 <= global_strip <= 255:
        return 9
    if 256 <= global_strip <= 383:
        return 10
    if 512 <= global_strip <= 639:
        return 12
    if 640 <= global_strip <= 767:
        return 13
    if 768 <= global_strip <= 895:
        return 14
    # Strips in 384-511 and 896-1023 are pass-through.
    return None


def apv_global_range(apv):
    start = (apv - 8) * 128
    return start, start + 127


def remap_strip(apv, strip):
    start, end = apv_global_range(apv)

    if start <= strip <= end:
        local = strip - start
    elif 0 <= strip <= 127:
        local = strip
    else:
        raise ValueError(
            "Strip {} is not in APV {}'s global range [{}, {}] "
            "and not a valid local index [0, 127].".format(strip, apv, start, end)
        )

    if apv in OUTER_APVS:
        local_remapped = 2 * local if local < 64 else 255 - 2 * local
    elif apv in MIDDLE_APVS:
        local_remapped = local + 64 if local < 64 else local - 64
    else:
        local_remapped = local

    global_remapped = start + local_remapped
    direction = "u" if global_remapped % 2 == 0 else "v"
    return global_remapped, direction


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------
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

        # Detector 5 is rotated 45°: strips are u (even remapped) or v (odd remapped).
        # Stored as (remapped_global_strip, [adc0..adcN]).
        self.detector_5_u = []
        self.detector_5_v = []

        self.id = ident

    def preprocess(self, strips, adcs, detector_ids, planes, adc_threshold):

        for i in range(len(strips)):
            if np.sum([adcs[j][i] for j in range(len(adcs))]) <= len(adcs) * adc_threshold:
                continue

            pulse = [adcs[j][i] for j in range(len(adcs))]
            data = (strips[i], pulse)

            match detector_ids[i]:
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
                case 5:
                    # Detector 5: ignore planeID, split on remapped-strip parity.
                    apv = global_to_apv(strips[i])
                    if apv is None:
                        # Pass-through range (384-511 / 896-1023) — skip for now,
                        # since those aren't remapped into u/v.
                        continue
                    remapped_strip, direction = remap_strip(apv, strips[i])
                    remapped_data = (remapped_strip, pulse)

                    # For APVs 12, 13, 14 (right edge), flip the u/v assignment
                    # relative to the parity-based direction.
                    if apv in RIGHT_APVS:
                        direction = "v" if direction == "u" else "u"

                    if direction == "u":
                        self.detector_5_u.append(remapped_data)
                    else:
                        self.detector_5_v.append(remapped_data)

    def empty(self):
        return (len(self.detector_0_x) + len(self.detector_0_y)
             + len(self.detector_1_x) + len(self.detector_1_y)
             + len(self.detector_2_x) + len(self.detector_2_y)
             + len(self.detector_3_x) + len(self.detector_3_y)
             + len(self.detector_5_u) + len(self.detector_5_v)
             <= 0)


# ---------------------------------------------------------------------------
# Preprocessing (unchanged except it now also picks up detID==5)
# ---------------------------------------------------------------------------
def preprocessing_batch(data, event_range, good_events, threshold, good_events_mutex,):
    adc_branches = [f"adc{i}" for i in range(15)]

    for event_id in range(event_range[0], event_range[1]):
        e = Event(event_id)

        strips = data["strip"][e.id]
        detids = data["detID"][e.id]
        planes = data["planeID"][e.id]
        adcs = [data[adc][e.id] for adc in adc_branches]

        e.preprocess(strips, adcs, detids, planes, threshold)

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

    for k in range(0, n // batch_size, 4):
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


# ---------------------------------------------------------------------------
# Hit pairing
# ---------------------------------------------------------------------------
def get_hits(detector_x, detector_y):
    """Pair each x-strip with its best-matching y-strip by raw dot product."""
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


# ---------------------------------------------------------------------------
# Detector 5 geometry
# ---------------------------------------------------------------------------
#
# Coordinate system (per user spec):
#   - Origin at top-left of the detector.
#   - x increases downward (top -> bottom), range [0, D].
#   - y increases rightward (left -> right), range [0, D].
#
# APV layout:
#   - APVs 8, 9, 10  sit on the LEFT edge  (y = 0).
#   - APVs 12, 13, 14 sit on the RIGHT edge (y = D).
#
# Pitch = 1 (unitless), D = 128.
#
# Anchor x-coordinate along the edge is:
#   - Left side: anchor_x = remapped_global_strip          (strip 0..383 -> x 0..383)
#   - Right side: anchor_x = remapped_global_strip - 512   (strip 512..895 -> x 0..383)
#
# Since the detector is D=128 tall but a single edge hosts 3*128 = 384 strips,
# the anchor_x values above go well beyond [0, 128] for most strips. That's
# fine — most (u, v) pairs will just intersect outside the detector and get
# filtered out. Only strips with anchors in [0, 128] can produce valid hits,
# but we don't assume that a priori — the geometry filter does the work.
#
# Strips run at +/-45 degrees from their anchor edge. We don't assume which
# sign corresponds to u vs v — instead we try both candidate directions for
# each strip and accept any direction choice that yields an intersection
# inside [0, D] x [0, D].
# ---------------------------------------------------------------------------

DETECTOR_SIZE = 384
LEFT_APVS  = {8,  9,  10}
RIGHT_APVS = {12, 13, 14}


def strip_geometry(remapped_strip):
    """
    Return (anchor, candidate_directions) for a remapped global strip.

    anchor : np.array shape (2,) in (x, y) where x is down, y is right.
    candidate_directions : list of two np.array shape (2,), unit length.

    The two candidate directions are the strip's two possible +/-45 deg
    orientations pointing into the detector interior.
    """
    apv = global_to_apv(remapped_strip)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    if apv in LEFT_APVS:
        # Left edge: y = 0, strip anchored along the x-axis.
        anchor = np.array([float(remapped_strip), 0.0])
        # Base inward direction is +y. +/-45 deg from that, in a frame where
        # x is down and y is right, gives these two unit vectors:
        #   "down-right": (+x, +y)/sqrt2
        #   "up-right"  : (-x, +y)/sqrt2
        candidates = [
            np.array([+inv_sqrt2, +inv_sqrt2]),
            np.array([-inv_sqrt2, +inv_sqrt2]),
        ]
    elif apv in RIGHT_APVS:
        # Right edge: y = D, strip anchored along the x-axis.
        anchor = np.array([float(remapped_strip - 512), float(DETECTOR_SIZE)])
        # Base inward direction is -y. +/-45 deg from that gives:
        #   "down-left": (+x, -y)/sqrt2
        #   "up-left"  : (-x, -y)/sqrt2
        candidates = [
            np.array([+inv_sqrt2, -inv_sqrt2]),
            np.array([-inv_sqrt2, -inv_sqrt2]),
        ]
    else:
        raise ValueError(f"Strip {remapped_strip} is not on a detector-5 edge.")

    return anchor, candidates


def compute_intersection(anchor_u, dir_u, anchor_v, dir_v):
    """
    Find where two infinite lines intersect, using the translate -> rotate ->
    solve -> pop-stack approach.

    Line U: p = anchor_u + t * dir_u
    Line V: p = anchor_v + s * dir_v

    Step 1: Translate by -anchor_u so U passes through origin.
    Step 2: Rotate by R = R(-theta_u) so dir_u aligns with the +x axis.
            In this frame, U is the x-axis; intersection occurs where V's
            y-coordinate is 0.
    Step 3: Solve for s on V such that (rotated V at s).y == 0.
    Step 4: t = (rotated V at s).x  (that's the x-coord of the intersection
            in the transformed frame, i.e. distance along U from its anchor).
    Step 5: The intersection point in the *original* frame is simply
            anchor_u + t * dir_u  (popping R^-1 and T^-1 amounts to this).

    Returns (intersection_point, t, s) or None if the lines are parallel.
    """
    # Rotation that sends dir_u -> (1, 0): this is R(-theta_u). Its matrix is
    #   [[ cos, sin],
    #    [-sin, cos]]  where (cos, sin) = dir_u components.
    cu, su = dir_u[0], dir_u[1]   # dir_u is already a unit vector
    R = np.array([[ cu,  su],
                  [-su,  cu]])

    # Transform anchor_v and dir_v into the U-aligned frame.
    anchor_v_rot = R @ (anchor_v - anchor_u)
    dir_v_rot    = R @ dir_v

    # Solve anchor_v_rot.y + s * dir_v_rot.y = 0 for s.
    if abs(dir_v_rot[1]) < 1e-12:
        return None  # V is parallel to U in the transformed frame -> parallel in original.
    s = -anchor_v_rot[1] / dir_v_rot[1]

    # x-coordinate of intersection in transformed frame = distance t along U.
    t = anchor_v_rot[0] + s * dir_v_rot[0]

    # Pop the stack: intersection in original frame is anchor_u + t * dir_u.
    intersection = anchor_u + t * dir_u
    return intersection, t, s


def inside_detector(point, tol=1e-9):
    """True if point lies inside [0, D] x [0, D] (inclusive of boundary)."""
    x, y = point
    return (-tol <= x <= DETECTOR_SIZE + tol) and (-tol <= y <= DETECTOR_SIZE + tol)


def find_valid_intersection(strip_u, strip_v):
    """
    Try all 4 sign combinations of (dir_u, dir_v) and return the first
    intersection that lands inside the detector, along with diagnostic info.

    Returns (intersection_point, chosen_dir_u, chosen_dir_v) or None.
    """
    anchor_u, dirs_u = strip_geometry(strip_u)
    anchor_v, dirs_v = strip_geometry(strip_v)

    valid_hits = []
    for du in dirs_u:
        for dv in dirs_v:
            result = compute_intersection(anchor_u, du, anchor_v, dv)
            if result is None:
                continue
            pt, t, s = result
            # Require t >= 0 and s >= 0: the strip only exists going INTO the
            # detector from its anchor edge, not behind it.
            if t < -1e-9 or s < -1e-9:
                continue
            if inside_detector(pt):
                valid_hits.append((pt, du, dv))

    if not valid_hits:
        return None
    # If multiple direction-sign choices yield inside-detector hits, take
    # the first. In practice with 45-deg strips on opposite edges this is
    # almost always unique.
    return valid_hits[0]


def get_det5_hits(detector_u, detector_v):
    """
    Pair each u-strip with its best-matching v-strip by raw dot product of
    their ADC pulses, but ONLY among v-strips whose geometric intersection
    with the u-strip lies inside the detector square [0, D] x [0, D].

    For each u-strip:
      1. Find the subset of v-strips that produce a valid intersection.
      2. Among those, pick the v-strip maximizing dot(u_pulse, v_pulse).
      3. Emit (u_hit, v_hit, intersection_point).

    Returns a list of tuples:
      ((u_strip, u_pulse), (v_strip, v_pulse), np.array([x, y]))
    """
    hit_ordered_pairs = []

    for hit_u in detector_u:
        u_strip, u_pulse = hit_u

        max_similarity = -np.inf
        best_pair = None
        best_point = None

        for hit_v in detector_v:
            v_strip, v_pulse = hit_v

            geom = find_valid_intersection(u_strip, v_strip)
            if geom is None:
                continue
            point, _du, _dv = geom

            similarity = np.dot(u_pulse, v_pulse)
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (hit_u, hit_v)
                best_point = point

        if best_pair is not None:
            hit_ordered_pairs.append((best_pair[0], best_pair[1], best_point))

    return hit_ordered_pairs


# ---------------------------------------------------------------------------
# Per-hit ADC helpers (unchanged)
# ---------------------------------------------------------------------------
def get_center_of_mass(adcs):
    """
    Time-bin center of mass of a pulse (15 ADC samples).

    Returns an integer time index in [0, len(adcs) - 1], or None if the
    pulse is entirely zero (can't compute a CoM).

    Uses int64 arithmetic to avoid overflow when ADC values are large.
    """
    weighted_sum = np.int64(0)
    total_mass = np.int64(0)

    for l, adc in enumerate(adcs):
        weighted_sum += np.int64(adc) * l
        total_mass += np.int64(adc)

    if total_mass <= 0:
        return None

    idx = int(np.round(weighted_sum / total_mass))
    # Clamp to valid range (rounding or negative ADCs could push it slightly out).
    return max(0, min(len(adcs) - 1, idx))


def get_hit_adc(adcs_x, adcs_y):
    """
    Return the ADC value to deposit in the heatmap for a (u, v) or (x, y)
    strip pair.

    Algorithm: find the center-of-mass time bin of each strip's pulse,
    look up each strip's ADC at its own CoM time bin, and return the max
    of the two values.
    """
    x_i = get_center_of_mass(adcs_x)
    y_i = get_center_of_mass(adcs_y)
    if x_i is None and y_i is None:
        return 0
    if x_i is None:
        return int(adcs_y[y_i])
    if y_i is None:
        return int(adcs_x[x_i])
    return int(max(adcs_x[x_i], adcs_y[y_i]))


# ---------------------------------------------------------------------------
# Beam reconstruction — detectors 0-3 (original) and detector 5 (new raw pairs)
# ---------------------------------------------------------------------------
def beam_reconstruction(root_file_path, threshold, detector):
    """
    For detectors 0-3: return a 128x128 ADC-weighted grid (original behavior).
    For detector 5: return a flat list of (u_hit, v_hit) pairs across all
        events — you said you want to handle the u/v -> physical geometry
        yourself, so no grid is built here.
    """
    good_events = preprocessing(root_file_path, threshold)

    if detector == 5:
        beam_reconstruction_grid = np.zeros((384, 384))
        for event in good_events:
            pairs = get_det5_hits(event.detector_5_u, event.detector_5_v)
            for u_hit, v_hit, point in pairs:
                # point is (x, y) with x = down-axis, y = right-axis.
                # Floor to integer cell, clamp to [0, 383].
                xi = int(np.floor(point[0]))
                yi = int(np.floor(point[1]))
                if xi < 0 or xi > 383 or yi < 0 or yi > 383:
                    # A point on the exact boundary (e.g. x=384.0) lands here;
                    # clamp rather than discard.
                    xi = min(max(xi, 0), 383)
                    yi = min(max(yi, 0), 383)
                # Mirror the row index so the plot's visual orientation matches
                # the physical detector when plot_beam_reconstruction uses
                # origin="lower".
                beam_reconstruction_grid[383 - xi, yi] += get_hit_adc(
                    u_hit[1], v_hit[1]
                )
        return beam_reconstruction_grid

    beam_reconstruction_grid = np.zeros((128, 128))

    for event in good_events:
        hits = []
        match detector:
            case 0:
                hits = get_hits(event.detector_0_x, event.detector_0_y)
            case 1:
                hits = get_hits(event.detector_1_x, event.detector_1_y)
            case 2:
                hits = get_hits(event.detector_2_x, event.detector_2_y)
            case 3:
                hits = get_hits(event.detector_3_x, event.detector_3_y)

        for hit in hits:
            if hit is not None:
                beam_reconstruction_grid[127 - hit[0][0], hit[1][0]] += get_hit_adc(hit[0][1], hit[1][1])

    return beam_reconstruction_grid


def plot_beam_reconstruction(grid, plot_name):
    # Scale ticks with grid size: 8 ticks across, rounded to a nice step.
    n = grid.shape[0]
    step = max(1, n // 8)
    ticks = np.arange(0, n + 1, step)
    plt.imshow(grid, cmap='viridis', origin="lower")
    plt.colorbar()
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(plot_name)
    plt.show()


# ---------------------------------------------------------------------------
# Example runs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Original detector-2 plot still works.
    # start_t = time()
    # grid = beam_reconstruction("RootFiles/nov8.root", 100, 2)
    # end_t = time()
    # print(f"time: {end_t - start_t}")
    # plot_beam_reconstruction(grid, "detector 2 adc threshold 100")

    # Detector 5: grid with intersection points weighted by get_hit_adc.
    start_t = time()
    det5_grid = beam_reconstruction("RootFiles/nov8.root", 10, 5)
    end_t = time()
    print(f"time: {end_t - start_t}")
    print(f"det 5 total ADC in grid: {det5_grid.sum():.0f}")
    print(f"det 5 nonzero cells:     {np.count_nonzero(det5_grid)}")
    plot_beam_reconstruction(det5_grid, "detector 5 adc threshold 100")

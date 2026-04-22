"""
Det 5 beam reconstruction -- per-side geometry.

Geometry (square sensor, W = 384 strip-pitch units):

    Left side  (APVs 8, 9, 10,  strips   0- 383):
        mapped strip s -> entry point (0, s)
        u strip (even raw) direction (+1, +1)
        v strip (odd  raw) direction (+1, -1)

    Right side (APVs 12, 13, 14, strips 512- 895):
        mapped strip s, re-indexed s' = s - 512 -> entry point (384, s')
        u strip (even raw) direction (-1, +1)
        v strip (odd  raw) direction (-1, -1)

    APVs 11 (384-511) and 15 (896-1023) are ignored.

    u_left  is parallel to v_right, and u_right to v_left, so the only
    valid stereo pairs are:
        u_left  x v_left   -> intersection in the LEFT triangle
        u_right x v_right  -> intersection in the RIGHT triangle

    Intersections:
        Left:   s_u, s_v at (0, s_u) and (0, s_v).
                Valid only if s_v >= s_u (lines meet inside the sensor).
                (x, y) = ( (s_v - s_u)/2,  (s_u + s_v)/2 )
        Right:  s_u', s_v' at (W, s_u') and (W, s_v').
                Valid only if s_v' >= s_u'.
                (x, y) = ( W - (s_v' - s_u')/2,  (s_u' + s_v')/2 )
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# APV layout
# ---------------------------------------------------------------------------

LEFT_APVS   = {8, 9, 10}
RIGHT_APVS  = {12, 13, 14}
OUTER_APVS  = {8, 10, 12, 14}
MIDDLE_APVS = {9, 13}

W = 384  # sensor width in strip-pitch units


def global_to_apv(g):
    if   0   <= g <= 127:  return 8
    elif 128 <= g <= 255:  return 9
    elif 256 <= g <= 383:  return 10
    elif 512 <= g <= 639:  return 12
    elif 640 <= g <= 767:  return 13
    elif 768 <= g <= 895:  return 14
    return None   # APV 11 / 15 -> ignored


def apv_global_range(apv):
    start = (apv - 8) * 128
    return start, start + 127


# ---------------------------------------------------------------------------
# Remap (validated against KhurramsRemap.xlsx for all APVs 8-14)
# ---------------------------------------------------------------------------

def remap_strip(apv, raw_strip):
    """
    Returns (mapped_global_strip, direction, side).
        direction: 'u' if raw even, 'v' if raw odd.
        side:      'left' for 8/9/10, 'right' for 12/13/14.
    """
    start, _ = apv_global_range(apv)
    local = raw_strip - start

    if apv in OUTER_APVS:
        local_mapped = 2 * local if local < 64 else 255 - 2 * local
    elif apv in MIDDLE_APVS:
        local_mapped = local + 64 if local < 64 else local - 64
    else:
        local_mapped = local   # not reached; kept for safety

    mapped    = start + local_mapped
    direction = 'u' if (raw_strip % 2 == 0) else 'v'
    side      = 'left' if apv in LEFT_APVS else 'right'
    return mapped, direction, side


# ---------------------------------------------------------------------------
# Intersections
# ---------------------------------------------------------------------------

def intersect_left(s_u, s_v):
    if s_v < s_u:
        return None
    x = (s_v - s_u) * 0.5
    y = (s_u + s_v) * 0.5
    if 0 <= x <= W and 0 <= y <= 2 * W:
        return x, y
    return None


def intersect_right(s_u_prime, s_v_prime):
    if s_v_prime < s_u_prime:
        return None
    x = W - (s_v_prime - s_u_prime) * 0.5
    y = (s_u_prime + s_v_prime) * 0.5
    if 0 <= x <= W and 0 <= y <= 2 * W:
        return x, y
    return None


# ---------------------------------------------------------------------------
# Event reconstruction
# ---------------------------------------------------------------------------

def reconstruct_event(left_u, left_v, right_u, right_v):
    """
    Each bucket: {mapped_strip_index: peak_adc}.
    Right-side keys are still the raw mapped values (512..895);
    we subtract 512 internally.
    Returns list of (amplitude, x, y).
    """
    out = []

    for su, pu in left_u.items():
        for sv, pv in left_v.items():
            xy = intersect_left(su, sv)
            if xy is not None:
                out.append((min(pu, pv), xy[0], xy[1]))

    for su, pu in right_u.items():
        for sv, pv in right_v.items():
            xy = intersect_right(su - 512, sv - 512)
            if xy is not None:
                out.append((min(pu, pv), xy[0], xy[1]))

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(points, bins=384, title='Det 5 Hit Heatmap'):
    if not points:
        print('No points to plot.')
        return
    pts = np.asarray(points, dtype=float)
    w, x, y = pts[:, 0], pts[:, 1], pts[:, 2]
    plt.figure(figsize=(8, 8))
    plt.hist2d(x, y, bins=bins, weights=w, cmap='inferno')
    plt.colorbar(label='Coincidence amplitude')
    plt.title(title)
    plt.xlabel('x (strip-pitch units)')
    plt.ylabel('y (strip-pitch units)')
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(root_file_path='RootFiles/nov8.root',
         peak_threshold=50,
         det_id=5):
    tree = uproot.open(root_file_path)['THit']
    branches = ['strip', 'evtID', 'detID'] + [f'adc{i}' for i in range(15)]
    data = tree.arrays(branches, library='np')

    all_hits = []
    n_events = len(data['evtID'])
    n_kept = 0

    for ev in range(n_events):
        strips = data['strip'][ev]
        detids = data['detID'][ev]
        adcs = np.stack([data[f'adc{i}'][ev] for i in range(15)], axis=1)

        left_u, left_v, right_u, right_v = {}, {}, {}, {}

        for i in range(len(strips)):
            if detids[i] != det_id:
                continue
            raw = int(strips[i])
            apv = global_to_apv(raw)
            if apv is None:
                continue
            peak = float(np.max(adcs[i]))
            if peak < peak_threshold:
                continue
            mapped, direction, side = remap_strip(apv, raw)
            bucket = {('left',  'u'): left_u,
                      ('left',  'v'): left_v,
                      ('right', 'u'): right_u,
                      ('right', 'v'): right_v}[(side, direction)]
            if peak > bucket.get(mapped, -np.inf):
                bucket[mapped] = peak

        if (left_u and left_v) or (right_u and right_v):
            hits = reconstruct_event(left_u, left_v, right_u, right_v)
            if hits:
                all_hits.extend(hits)
                n_kept += 1

    print(f'Processed {n_events} events, kept {n_kept} with reconstructable hits.')
    print(f'Total intersections: {len(all_hits)}')
    plot_heatmap(all_hits)
    return all_hits


if __name__ == '__main__':
    main()

import sys
import uproot
import queue
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from threading import Thread, Lock


# multithreading global constants
THREAD_POOL_SIZE: int = 4
EVENT_BATCH_SIZE: int = 2000
MAX_QUEUE_SIZE = 256
JOB_QUEUE = queue.Queue(MAX_QUEUE_SIZE)

# detector geometric information
LEFT_SIDE_XPOS: int = 0
RIGHT_SIDE_XPOS: int = 384

# program parameters
DETECTOR = 5
ADC_THRESHOLD = 500 * 15
PLANE_X = 0
PLANE_Y = 1

# apv information
APV_OFFSET: int = 128
APV_WHITELIST: list[int] = [0, 1, 2]

# root file useful branches
TREE_NAME = "THit"
BRANCHES: list[str] = ["evtID", "detID", "planeID", "strip"] + [f"adc{i}" for i in range(15)]

# matplotlib visualization variables
GRID = np.zeros((768, 768))
GRID_LOCK = Lock()
PLANAR_GRID = np.zeros((384, 384))

def get_apv(strip: int) -> int:
    return strip // APV_OFFSET

def get_xy(apv_a: int, 
           apv_b: int, 
           mapped_strip_a: int, 
           mapped_strip_b: int, 
           dir_a: npt.NDArray[np.int32], 
           dir_b: npt.NDArray[np.int32]
) -> tuple[int, int] | None:

        D = np.array([
            [dir_a[0], -dir_b[0]],
            [dir_a[1], -dir_b[1]]
        ])

        xa = LEFT_SIDE_XPOS if apv_a < 4 else RIGHT_SIDE_XPOS
        xb = LEFT_SIDE_XPOS if apv_b < 4 else RIGHT_SIDE_XPOS
        ya = mapped_strip_a
        yb = mapped_strip_b

        p = np.array([xb - xa, yb - ya])

        try:
            s_t = np.linalg.solve(D, p)
        except np.linalg.LinAlgError:
            return None

        l1 = np.array([xa, ya]) + s_t[0] * dir_a 
        l2 = np.array([xb, yb]) + s_t[1] * dir_b 
        assert np.allclose(l1, l2)

        return l1 * 2
     

def job_planar(df, start, end):
    for event in range(start, end):
        adcs = [df[f"adc{i}"][event] for i in range(15)]
        strips = df["strip"][event]
        dets = df["detID"][event]
        planes = df["planeID"][event]
        event_hit_amount = len(strips)

        det_indices = [i for i in range(event_hit_amount) if dets[i] == DETECTOR]

        if len(det_indices) <= 0:
            continue

        planes_x_indices = []
        planes_y_indices = []

        for x in det_indices:
            if planes[x] == PLANE_X:
                planes_x_indices.append(x)
            else:
                planes_y_indices.append(x)

        if len(planes_x_indices) <= 0 or len(planes_y_indices) <= 0:
            continue

        for i in planes_x_indices:
            for j in planes_y_indices:
                x = strips[i]
                y = strips[j]
                adcs_i = [adcs[k][i] for k in range(15)]
                adcs_j = [adcs[k][j] for k in range(15)]

                if sum(adcs_i) < ADC_THRESHOLD  and sum(adcs_j) < ADC_THRESHOLD:
                    continue

                adc_max = max(adcs_i + adcs_j)

                GRID_LOCK.acquire()
                PLANAR_GRID[x, y] += adc_max 
                GRID_LOCK.release()

# TODO
def job_cylindrical(df, start, end):
    for event in range(start, end):
        adcs = [df[f"adc{i}"][event] for i in range(15)]
        strips = df["strip"][event]
        dets = df["detID"][event]
        planes = df["planeID"][event]
        event_hit_amount = len(strips)

        det_indices = [i for i in range(event_hit_amount) if dets[i] == DETECTOR]

        if len(det_indices) <= 0:
            continue

        planes_x_indices = []
        planes_y_indices = []

        for x in det_indices:
            if planes[x] == PLANE_X:
                planes_x_indices.append(x)
            else:
                planes_y_indices.append(x)

        if len(planes_x_indices) <= 0 or len(planes_y_indices) <= 0:
            continue

        for i in planes_x_indices:
            for j in planes_y_indices:
                x = strips[i]
                y = strips[j]
                adcs_i = [adcs[k][i] for k in range(15)]
                adcs_j = [adcs[k][j] for k in range(15)]

                if sum(adcs_i) < ADC_THRESHOLD  and sum(adcs_j) < ADC_THRESHOLD:
                    continue

                adc_max = max(adcs_i + adcs_j)

                GRID_LOCK.acquire()
                PLANAR_GRID[x, y] += adc_max 
                GRID_LOCK.release()

def thread_loop():
    while True:
        current_job = JOB_QUEUE.get()

        try:
            if current_job is None:
                break

            f, args = current_job
            f(*args)
        finally:
            JOB_QUEUE.task_done()


def main():
    if len(sys.argv) != 2:
        print("usage: python left_side_det5.py <root file>")
        return

    filepath = sys.argv[1]

    file = uproot.open(filepath)
    tree = file[TREE_NAME]
    df = tree.arrays(BRANCHES, library="np")

    event_amount: int = len(df["evtID"])

    thread_pool: list[Thread] = [Thread(target=thread_loop) for _ in range(THREAD_POOL_SIZE)]
    for t in thread_pool:
        t.start()

    for k in range(0, event_amount, EVENT_BATCH_SIZE):
        print(k)

        start = k
        end = min(start + EVENT_BATCH_SIZE, event_amount)

        JOB_QUEUE.put((job, (df, start, end)))

    JOB_QUEUE.join()

    for _ in thread_pool:
        JOB_QUEUE.put(None)

    for t in thread_pool:
        t.join()

    plt.imshow(PLANAR_GRID)
    plt.show()

if __name__ == "__main__":
    main()

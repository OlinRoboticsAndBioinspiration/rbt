import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def quality(rbt, dbg=False, **kwds):
    t = rbt.t; X = rbt.X; N,_ = X.shape; j = rbt.j; u = rbt.u
    #x = X[..., j['x']]
    #y = X[..., j['y']]
    x = rbt.stand_x
    y = rbt.stand_y
    dx = X[..., j['dx']]
    dy = X[..., j['dy']]

    # apply the crop
    t = t[rbt.start_trial: rbt.stop_trial]
    x = x[rbt.start_trial: rbt.stop_trial]
    y = y[rbt.start_trial: rbt.stop_trial]

    dx = dx[rbt.start_trial: rbt.stop_trial]
    dy = dy[rbt.start_trial: rbt.stop_trial]

    speed = np.sqrt(dx * dx + dy * dy)

    #quality is going to be some metric of stability of speed

    max_speed = np.median(speed)
    def quality_on_window(segment):
        qual = 1.0 - np.clip(np.std(segment), 0, 1)
        return qual

    def rolling(series, func, win):
        num_curva = len(series)
        output = np.zeros((num_curva - win*2, 1))
        for i in range(win, num_curva -win):
            segment = series[i-win:i+win]
            output[i-win] = func(segment)
        return output

    window_size = 20
    speed = np.pad(speed, (window_size, window_size), mode="edge")
    quality = rolling(speed, quality_on_window, window_size)
    rbt.quality = quality

    #plt.plot(t, speed)
    #plt.plot(t, res)
    if dbg==True:
        plt.scatter(x, y, c=quality, lw=0, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('equal')
        plt.show()
    #raw_input()


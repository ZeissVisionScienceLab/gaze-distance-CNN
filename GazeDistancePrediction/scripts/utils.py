import numpy as np
from . import params

max = 20

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def pixeldepth2distance(p, cgl):
    """
    Function to convert a pixel to a depth value
    :param p: pixel coordinates
    :param cgl: combined gaze local coordinates
    :return: depth value
    """
    
    # convert i,j as matrix coordinates to x,y in image pixel coordinates (origin at bottom left)
    p_x = p[1]
    p_y = params.kernel_height - p[0]
    p_z = p[2]

    cgl_x = cgl[0]
    cgl_y = cgl[1]
    cgl_z = cgl[2]

    # 1. cgl to kernel image plane
    g_x = cgl_x / cgl_z
    g_y = cgl_y / cgl_z

    # 2. g to depth texture plane
    x = (p_x - params.kernel_height / 2) * np.tan(np.radians(params.kernel_fov / 2))/(params.kernel_height/2) + g_x
    y = (p_y - params.kernel_height / 2) * np.tan(np.radians(params.kernel_fov / 2))/(params.kernel_height/2) + g_y

    # 3. 2d to 3d point
    X = x * p_z
    Y = y * p_z
    Z = p_z

    # 4. 3d point to depth
    depth = np.sqrt(X**2 + Y**2 + Z**2)

    return depth

def get_random_point_in_circle(center, radius):
    """
    Function to get a random point inside a circle.

    Parameters
    ----------
        center: float
            center of the circle
        radius: float
            radius of the circle
    """

    r = radius * np.sqrt(np.random.rand())
    theta = np.random.rand() * 2 * np.pi
    x = int(r * np.cos(theta) + center[0])
    y = int(r * np.sin(theta) + center[1])

    return np.array([x, y])

def normalize_depth(depth):
    """
    Function to normalize depth values to [0,1].

    Parameters
    ----------
    :param depth: depth value
    :return: normalized depth value

    """
    
    depth = np.clip(depth, 0, max)
    return depth / max
    #return depth / params.far_clipping_plane

def denormalize_depth(depth):
    """
    Function to recieve actual distances again

    Parameters
    ----------
    :param depth: normalized depth value in [0,1]
    :return: actual depth value
    """
    return depth * max #* params.far_clipping_plane

def normalize(x: np.array):
    """
    Function to normalize values to [0,1].

    Parameters
    ----------
    x: np.array
        values to normalize
    return: 
        normalized value
    """
    min = np.min(x)
    max = np.max(x)

    x = (x - min) / (max - min)

    return x
    
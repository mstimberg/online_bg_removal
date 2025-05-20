import numpy as np
import pandas as pd


def segments_from_table(table):
    # Extracts a list of uninterrupted segments from a table
    # Copies are returned(not views).
    segments = []
    for _, traj in table.groupby('id', sort=True):
        breaks = list(1+(np.diff(traj['frame'] ) > 1).nonzero()[0]) + [len(traj)]
        n = 0
        for i in breaks:
            segment = pd.DataFrame(traj.iloc[n:i])
            if len(segment) > 2:  # otherwise we can't do any calculation
                segments.append(segment)
            n = i
    return segments


def calculate_features(segment):
    '''
    Calculate trajectory features for a contiguous segment:
    - vx (velocity)
    - vy
    - speed
    - ax (acceleration)
    - ay
    - motion_reversal (dot product between two successive velocity vectors)
    - reversal (dot product between motion vector and angle)
    - angular speed (angle is modulo pi)
    '''
    if len(segment)>2:
        vx, vy = np.diff(segment['x']), np.diff(segment['y'])
        segment['vx'] = np.hstack([vx, np.nan])
        segment['vy'] = np.hstack([vy, np.nan])
        speed = (vx**2 + vy**2)**.5
        segment['speed'] = np.hstack([speed, np.nan])

        segment['ax'] = np.hstack([np.nan, np.diff(np.diff(segment['x'])), np.nan])
        segment['ay'] = np.hstack([np.nan, np.diff(np.diff(segment['y'])), np.nan])
        segment['acceleration'] = (segment['ax']**2 + segment['ay']**2)**.5

        segment['motion_reversal'] = np.hstack([np.nan, (vx[1:]*vx[:-1] + vy[1:]*vy[:-1])/ (speed[1:]*speed[:-1]), np.nan])
        segment['reversal'] = segment['vx']*np.cos(segment['angle']) + segment['vy']*np.sin(segment['angle'])

        angle = segment['angle']
        angular_speed = ((np.diff(angle) + np.pi / 2) % np.pi) - np.pi / 2
        segment['angular_speed'] = np.hstack([ angular_speed, np.nan])
    return segment

def mark_avoiding_reactions_from_motion(segment, refractory=1, threshold=None, dt=1.):
    '''
    Mark the start of an avoiding reaction, calculated from motion.

    Refractoriness not taken into account here.
    '''
    if threshold is None:
        threshold_cos = 0
    else:
        threshold_cos = np.cos(threshold*dt) # maximum dot product
    backward_swimming = segment['motion_reversal']<threshold_cos
    segment['mAR_start'] = np.hstack([np.diff(1*backward_swimming)>0, False])
    return segment
import os
import shutil
import numpy as np

from scipy.ndimage import zoom
from skimage.data import camera
from scipy.spatial.distance import cdist


def duration_to_alignment(in_duration):
  total_len = np.sum(in_duration)
  num_chars = len(in_duration)

  attention = np.zeros(shape=(num_chars,total_len),dtype=np.float32)
  y_offset = 0

  for duration_idx, duration_val in enumerate(in_duration):
    for y_val in range(0,duration_val):
      attention[duration_idx][y_offset + y_val] = 1.0
    
    y_offset += duration_val
  
  return attention


def rescale_alignment(in_alignment,in_targcharlen):
  current_x = in_alignment.shape[0]
  x_ratio = in_targcharlen / current_x
  pivot_points = []
  
  zoomed = zoom(in_alignment,(x_ratio,1.0),mode="nearest")

  for x_v in range(0,zoomed.shape[0]):
    for y_v in range(0,zoomed.shape[1]):
      val = zoomed[x_v][y_v]
      if val < 0.5:
        val = 0.0
      else:
        val = 1.0
        pivot_points.append( (x_v,y_v) )

      zoomed[x_v][y_v] = val
      
  
  if zoomed.shape[0] != in_targcharlen:
    print("Zooming didn't rshape well, explicitly reshaping")
    zoomed.resize((in_targcharlen,in_alignment.shape[1]))

  return zoomed, pivot_points

    

def get_pivot_points(in_att):
  ret_points = []
  for x in range(0, in_att.shape[0]):
    for y in range(0, in_att.shape[1]):
      if in_att[x,y] > 0.8:
        ret_points.append((x,y))
  return ret_points

def compensate_duration_mel(in_durs,in_mel_len,in_text_len,in_audiofn):
    durations_len = np.sum(in_durs)
    durations_n = in_durs
    
    if in_mel_len > durations_len:
        durations_n[-1] += in_mel_len - durations_len
    else:
        if durations_len > in_mel_len:
            diff = durations_len - in_mel_len
            found = False
            for r in reversed(range(len(durations_n) - 1)):
                if diff == 0:
                    found = True
                    break
                    
                if durations_n[r] >= diff:
                    durations_n[r] -= diff
                    found = True
                    break
                else:
                    diff -= durations_n[r]
                    durations_n[r] = 0
                    
            if not found:
                raise RuntimeError(f"Oh nonono MFABros what happened!!!!!!!????? (Failed to compensate for diff in durations and mel len for file {in_audiofn})")
                
                
    return durations_n

    
def gather_dist(in_mtr,in_points):
  #initialize with known size for fast
  full_coords = [(0,0) for x in range(in_mtr.shape[0] * in_mtr.shape[1])]
  i = 0
  for x in range(0, in_mtr.shape[0]):
    for y in range(0, in_mtr.shape[1]):
      full_coords[i] = (x,y)
      i += 1
  
  return cdist(full_coords, in_points,"euclidean")
    
def create_guided(in_align,in_pvt,looseness):
  new_att = np.ones(in_align.shape,dtype=np.float32)
  # It is dramatically faster that we first gather all the points and calculate than do it manually
  # for each point in for loop
  dist_arr = gather_dist(in_align,in_pvt)
  # Scale looseness based on attention size. (addition works better than mul). Also divide by 100
  # because having user input 3.35 is nicer
  real_loose = (looseness / 100) * (new_att.shape[0] + new_att.shape[1])
  g_idx = 0
  for x in range(0, new_att.shape[0]):
    for y in range(0, new_att.shape[1]):
      min_point_idx = dist_arr[g_idx].argmin()

      closest_pvt = in_pvt[min_point_idx]
      distance = dist_arr[g_idx][min_point_idx] / real_loose
      distance = np.power(distance,2) 

      g_idx += 1
      
      new_att[x,y] = distance

  return np.clip(new_att,0.0,1.0)

def durations_to_mask(in_durs,in_mel_len,in_text_len,in_audiofn,in_loose):
    
    duration_arr = compensate_duration_mel(in_durs,in_mel_len,in_text_len,in_audiofn)
    align = duration_to_alignment(duration_arr)

    id_true_size = in_text_len
        
    if align.shape[0] != id_true_size:
        align, points = rescale_alignment(align,id_true_size)
    else:
        points = get_pivot_points(align)
    
    mask = create_guided(align,points,in_loose)
    return mask
    

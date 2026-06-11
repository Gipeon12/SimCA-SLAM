
# Odometric pose and scan values at time t
float32    sim_time                # t <s>
float32[]  odom_pose               # [x <m>, y <m>, yaw <rad>]
float32[]  scan_angles             # [ang <rad>  * N]   (N scans)
float32[]  scan_ranges             # [rge <m>    * N]   (N scans)
float32    laser_limit             # l <m>
string     status                  # "free_roaming", "busy" or "idle"

# Mutual observations (in body frame): names and polar coordinates of the observed agents
string[]   rob_names
float32[]  obs_angles
float32[]  obs_ranges

# Virtual landmark observation (in body frame): cartesian coordinates of the polygon vertices
float32[]  vlm_xs
float32[]  vlm_ys

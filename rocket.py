import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io, base64

# -------------------------------
#  GLOBAL PARAMETERS & CONSTANTS
# -------------------------------
dry_weight_stage1 = 3.5e5 * 0.45359   # lbs -> kg
propload_stage1   = 7.5e6 * 0.45359
dry_weight_stage2 = 2e5   * 0.45359
propload_stage2   = 3.3e6 * 0.45359

mass_stage2_total = dry_weight_stage2 + propload_stage2
mass_total        = mass_stage2_total + dry_weight_stage1 + propload_stage1

diameter      = 9.0
cross_section = np.pi * (diameter/2)**2
drag_coef     = 0.35

v_exhaust_stage1 = 3400
v_exhaust_stage2 = 3720
mdot_stage1       = 33 * 808
mdot_stage2       = (3*805) + (3*808)
burntime_stage1   = (propload_stage1*0.9) / mdot_stage1
burntime_stage2   = propload_stage2  / mdot_stage2

GM             = 3.986004418e14
radius_earth   = 6.371e6
rho            = 1.225
scale_height   = 8500
iss_altitude   = 420e3
iss_vel_orb    = math.sqrt(GM/(radius_earth + iss_altitude))

# For Cape Canaveral, Earth rotation, etc.
cape_lat_deg = 28.5
cape_lon_deg = -80.5
cape_lat = math.radians(cape_lat_deg)
cape_lon = math.radians(cape_lon_deg)
x0 = radius_earth * math.cos(cape_lat) * math.cos(cape_lon)
y0 = radius_earth * math.cos(cape_lat) * math.sin(cape_lon)
z0 = radius_earth * math.sin(cape_lat)

def earth_rotation_velocity(x, y, z):
    """Return local linear velocity due to Earth's rotation at (x, y, z)."""
    earth_omega = 7.2921159e-5  # rad/s
    w = np.array([0, 0, earth_omega])
    r_vec = np.array([x, y, z])
    return np.cross(w, r_vec)

# Timestep
dt = 0.1

# ---------------------------------
#   MAIN SIMULATION FUNCTION
# ---------------------------------
def simulate_gravity_turn_3d(vert_v0, perc_burntime2, coast_time):
    """
    3D multi-stage gravity turn from Cape Canaveral in ECI coords.
    Returns a DataFrame with columns: time, x, y, z, r, vel_x, vel_y, vel_z, etc.
    We skip any interactive widget usage here, just do the math.
    """
    pos0  = np.array([x0, y0, z0])
    vel0  = earth_rotation_velocity(*pos0)
    mass0 = mass_total

    def get_local_horizontal_unit(r_vec):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        ref = np.array([0,0,1])
        cross_ = np.cross(ref, r_hat)
        if np.linalg.norm(cross_) < 1e-10:
            cross_ = np.cross(np.array([1,0,0]), r_hat)
        perp_hat = cross_ / np.linalg.norm(cross_)
        return perp_hat

    def compute_thrust_direction(r_vec, v_vec, pitch_limit_deg, vert_v_target, stage_thrust):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        v_radial = np.dot(v_vec, r_hat)
        v_horiz_vec = v_vec - v_radial * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz_vec)
        gamma = math.atan2(v_radial, v_horiz_mag)

        # If radial velocity is below a certain threshold, force pitch = 0
        if v_radial < vert_v_target:
            pitch_rad = 0.0
        else:
            g_local = GM / (r**2)
            ratio   = (g_local / stage_thrust) * math.cos(gamma)
            ratio   = np.clip(ratio, -1, 1)  # avoid invalid acos
            pitch_rad = math.acos(ratio) - gamma
            pitch_rad = np.clip(pitch_rad, 0, math.radians(pitch_limit_deg))

        if v_horiz_mag > 1e-10:
            perp_hat = v_horiz_vec / v_horiz_mag
        else:
            # fallback
            perp_hat = get_local_horizontal_unit(r_vec)

        thrust_dir = (math.cos(pitch_rad)*r_hat +
                      math.sin(pitch_rad)*perp_hat)
        return thrust_dir

    def compute_thrust_direction_stage2(r_vec, v_vec, pitch_limit_deg, stage_thrust):
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        v_radial = np.dot(v_vec, r_hat)
        v_horiz_vec = v_vec - v_radial * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz_vec)
        gamma = math.atan2(v_radial, v_horiz_mag)
        altitude = r - radius_earth
        g_local  = GM / (r**2)

        # If altitude is large, pitch up near 85°, else use partial approach
        if altitude >= 1e6:
            pitch_rad = math.radians(85)
        else:
            ratio = (g_local / stage_thrust)*math.cos(gamma)
            ratio = np.clip(ratio, -1, 1)
            pitch_rad = math.acos(ratio) - gamma
            pitch_rad = np.clip(pitch_rad, 0, math.radians(pitch_limit_deg))

        if v_horiz_mag > 1e-10:
            perp_hat = v_horiz_vec / v_horiz_mag
        else:
            perp_hat = get_local_horizontal_unit(r_vec)

        thrust_dir = (math.cos(pitch_rad)*r_hat +
                      math.sin(pitch_rad)*perp_hat)
        return thrust_dir

    # --- Stage 1 ---
    n_steps_stage1 = int(burntime_stage1 / dt)
    t_stage1       = np.zeros(n_steps_stage1 + 1)
    pos_stage1     = np.zeros((n_steps_stage1+1, 3))
    vel_stage1     = np.zeros((n_steps_stage1+1, 3))
    acc_stage1     = np.zeros((n_steps_stage1+1, 3))
    mass_stage1_arr= np.zeros(n_steps_stage1+1)

    t_stage1[0]       = 0.0
    pos_stage1[0]     = pos0
    vel_stage1[0]     = vel0
    mass_stage1_arr[0]= mass0
    crashed = False

    for i in range(n_steps_stage1):
        t_stage1[i+1] = t_stage1[i] + dt
        r_vec = pos_stage1[i]
        v_vec = vel_stage1[i]
        r = np.linalg.norm(r_vec)
        if r <= radius_earth - 10.0:
            crashed = True
            break
        if t_stage1[i] < burntime_stage1:
            thrust_mag = (v_exhaust_stage1*mdot_stage1)/mass_stage1_arr[i]
        else:
            thrust_mag = 0.0
        thrust_dir = compute_thrust_direction(r_vec, v_vec,
                                              70,
                                              vert_v0,
                                              thrust_mag)
        thrust_acc = thrust_mag * thrust_dir
        g_local = GM / (r**2)
        grav_acc = -g_local * (r_vec/r)
        speed   = np.linalg.norm(v_vec)
        alt     = r - radius_earth

        if alt < 120e3:
            air_density = rho * np.exp(-alt/scale_height)
        else:
            air_density = 0.0
        drag_acc = -0.5 * (air_density*drag_coef*cross_section / mass_stage1_arr[i]) * speed * v_vec
        acc_stage1[i] = thrust_acc + grav_acc + drag_acc
        vel_stage1[i+1] = v_vec + acc_stage1[i]*dt
        pos_stage1[i+1] = r_vec + v_vec*dt + 0.5*acc_stage1[i]*(dt**2)

        if thrust_mag > 0.0:
            mass_stage1_arr[i+1] = mass_stage1_arr[i] - mdot_stage1*dt
        else:
            mass_stage1_arr[i+1] = mass_stage1_arr[i]

    df_stage1 = pd.DataFrame({
        'stage':  ['stage1']*(n_steps_stage1+1),
        'time':   t_stage1,
        'x':      pos_stage1[:,0],
        'y':      pos_stage1[:,1],
        'z':      pos_stage1[:,2],
        'r':      np.linalg.norm(pos_stage1, axis=1),
        'vel_x':  vel_stage1[:,0],
        'vel_y':  vel_stage1[:,1],
        'vel_z':  vel_stage1[:,2],
        'vel_orb':np.linalg.norm(vel_stage1, axis=1),
        'acc_x':  acc_stage1[:,0],
        'acc_y':  acc_stage1[:,1],
        'acc_z':  acc_stage1[:,2],
        'mass':   mass_stage1_arr
    })

    # --- Stage 2 (partial) ---
    pos_stage2_0 = pos_stage1[-1]
    vel_stage2_0 = vel_stage1[-1]
    burntime_stage2_part = burntime_stage2 * perc_burntime2
    n_steps_stage2 = int(burntime_stage2_part / dt)

    t_stage2   = np.zeros(n_steps_stage2+1)
    pos_stage2 = np.zeros((n_steps_stage2+1, 3))
    vel_stage2 = np.zeros((n_steps_stage2+1, 3))
    acc_stage2 = np.zeros((n_steps_stage2+1, 3))
    mass_stage2_arr = np.zeros(n_steps_stage2+1)

    t_stage2[0]     = t_stage1[-1]
    pos_stage2[0]   = pos_stage2_0
    vel_stage2[0]   = vel_stage2_0
    mass_stage2_arr[0] = mass_stage2_total

    for i in range(n_steps_stage2):
        t_stage2[i+1] = t_stage2[i] + dt
        r_vec = pos_stage2[i]
        v_vec = vel_stage2[i]
        r     = np.linalg.norm(r_vec)
        if r <= radius_earth:
            crashed=True
            break
        time_into_burn = t_stage2[i] - t_stage2[0]
        if time_into_burn < burntime_stage2_part:
            thrust_mag = (v_exhaust_stage2 * mdot_stage2)/mass_stage2_arr[i]
        else:
            thrust_mag = 0.0
        thrust_dir = compute_thrust_direction_stage2(r_vec, v_vec, 90,
                                                          thrust_mag)
        thrust_acc = thrust_mag * thrust_dir
        g_local = GM/(r**2)
        grav_acc = -g_local*(r_vec/r)
        speed = np.linalg.norm(v_vec)
        alt   = r - radius_earth

        if alt<120e3:
            air_density = rho*np.exp(-alt/scale_height)
        else:
            air_density = 0.0
        drag_acc = -0.5*(air_density*drag_coef*cross_section/mass_stage2_arr[i])*speed*v_vec

        acc_stage2[i] = thrust_acc + grav_acc + drag_acc
        vel_stage2[i+1] = v_vec + acc_stage2[i]*dt
        pos_stage2[i+1] = r_vec + v_vec*dt + 0.5*acc_stage2[i]*(dt**2)

        if thrust_mag>0.0:
            mass_stage2_arr[i+1] = mass_stage2_arr[i] - mdot_stage2*dt
        else:
            mass_stage2_arr[i+1] = mass_stage2_arr[i]

    df_stage2 = pd.DataFrame({
        'stage':  ['stage2']*(n_steps_stage2+1),
        'time':   t_stage2,
        'x':      pos_stage2[:,0],
        'y':      pos_stage2[:,1],
        'z':      pos_stage2[:,2],
        'r':      np.linalg.norm(pos_stage2, axis=1),
        'vel_x':  vel_stage2[:,0],
        'vel_y':  vel_stage2[:,1],
        'vel_z':  vel_stage2[:,2],
        'vel_orb':np.linalg.norm(vel_stage2, axis=1),
        'acc_x':  acc_stage2[:,0],
        'acc_y':  acc_stage2[:,1],
        'acc_z':  acc_stage2[:,2],
        'mass':   mass_stage2_arr
    })

    # --- Coast ---
    coast_steps = int(coast_time/dt)
    t_coast   = np.zeros(coast_steps+1)
    pos_coast = np.zeros((coast_steps+1, 3))
    vel_coast = np.zeros((coast_steps+1, 3))
    acc_coast = np.zeros((coast_steps+1, 3))
    mass_coast= np.zeros(coast_steps+1)

    t_coast[0]   = t_stage2[-1]
    pos_coast[0] = pos_stage2[-1]
    vel_coast[0] = vel_stage2[-1]
    mass_coast[0]= mass_stage2_arr[-1]

    for i in range(coast_steps):
        t_coast[i+1] = t_coast[i] + dt
        r_vec = pos_coast[i]
        r     = np.linalg.norm(r_vec)
        if r<=radius_earth:
            crashed=True
            break
        grav_acc = -GM/(r**2)*(r_vec/r)
        acc_coast[i] = grav_acc
        vel_coast[i+1] = vel_coast[i] + grav_acc*dt
        pos_coast[i+1] = pos_coast[i] + vel_coast[i]*dt + 0.5*grav_acc*(dt**2)
        mass_coast[i+1]= mass_coast[i]

    df_coast = pd.DataFrame({
        'stage':  ['coast']*(coast_steps+1),
        'time':   t_coast,
        'x':      pos_coast[:,0],
        'y':      pos_coast[:,1],
        'z':      pos_coast[:,2],
        'r':      np.linalg.norm(pos_coast, axis=1),
        'vel_x':  vel_coast[:,0],
        'vel_y':  vel_coast[:,1],
        'vel_z':  vel_coast[:,2],
        'vel_orb':np.linalg.norm(vel_coast, axis=1),
        'acc_x':  acc_coast[:,0],
        'acc_y':  acc_coast[:,1],
        'acc_z':  acc_coast[:,2],
        'mass':   mass_coast
    })

    # --- Stage 3 (final partial Stage 2) ---
    burntime_stage2_part2 = 0.65*(burntime_stage2*(1-perc_burntime2))
    n_steps_stage3 = int(burntime_stage2_part2/dt + 2*3600/dt)
    t_stage3   = np.zeros(n_steps_stage3+1)
    pos_stage3 = np.zeros((n_steps_stage3+1, 3))
    vel_stage3 = np.zeros((n_steps_stage3+1, 3))
    acc_stage3 = np.zeros((n_steps_stage3+1, 3))
    mass_stage3= np.zeros(n_steps_stage3+1)

    t_stage3[0]   = t_coast[-1]
    pos_stage3[0] = pos_coast[-1]
    vel_stage3[0] = vel_coast[-1]
    mass_stage3[0]= mass_coast[-1]

    for i in range(n_steps_stage3):
        t_stage3[i+1] = t_stage3[i] + dt
        r_vec = pos_stage3[i]
        v_vec = vel_stage3[i]
        r = np.linalg.norm(r_vec)
        if r<=radius_earth:
            crashed=True
            break
        grav_acc = -GM/(r**2)*(r_vec/r)
        time_in_stage3 = t_stage3[i] - t_stage3[0]
        if time_in_stage3 < burntime_stage2_part2:
            thrust_mag = (v_exhaust_stage2*mdot_stage2)/mass_stage3[i]
            # Force thrust direction to local horizontal for final insertion
            perp_hat = get_local_horizontal_unit(r_vec)
            thrust_acc = thrust_mag*perp_hat
            mass_stage3[i+1] = mass_stage3[i] - mdot_stage2*dt
        else:
            thrust_acc = np.array([0,0,0])
            mass_stage3[i+1] = mass_stage3[i]

        acc_stage3[i] = grav_acc + thrust_acc
        vel_stage3[i+1] = v_vec + acc_stage3[i]*dt
        pos_stage3[i+1] = r_vec + v_vec*dt + 0.5*acc_stage3[i]*(dt**2)

    df_stage3 = pd.DataFrame({
        'stage':  ['stage2_part2']*(n_steps_stage3+1),
        'time':   t_stage3,
        'x':      pos_stage3[:,0],
        'y':      pos_stage3[:,1],
        'z':      pos_stage3[:,2],
        'r':      np.linalg.norm(pos_stage3, axis=1),
        'vel_x':  vel_stage3[:,0],
        'vel_y':  vel_stage3[:,1],
        'vel_z':  vel_stage3[:,2],
        'vel_orb':np.linalg.norm(vel_stage3, axis=1),
        'acc_x':  acc_stage3[:,0],
        'acc_y':  acc_stage3[:,1],
        'acc_z':  acc_stage3[:,2],
        'mass':   mass_stage3
    })

    # Concatenate
    df = pd.concat([df_stage1, df_stage2, df_coast, df_stage3], ignore_index=True)
    return df

# --------------------------------------------
#   FUNCTION TO PLOT 3 VERTICAL SUBPLOTS
# --------------------------------------------
def plot_combined_figure(df):
    """
    Creates a combined figure with:
      - A 3D trajectory plot on the left.
      - Three vertical subplots on the right:
           1) Altitude vs Time
           2) Velocity Components vs Time
           3) Orbital Velocity vs Time
    Returns a Base64-encoded <img> string.
    """
    # Create a figure with a grid layout: left column for 3D plot, right column for 3 subplots.
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
    gs.update(hspace=0.5)  # Increase vertical spacing

    # -------------------------------
    # Left Column: 3D Trajectory Plot
    # -------------------------------
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.set_title("3D Trajectory from Cape Canaveral")

    # Plot Earth as a sphere
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    xsph = radius_earth * np.outer(np.sin(u), np.cos(v))
    ysph = radius_earth * np.outer(np.sin(u), np.sin(v))
    zsph = radius_earth * np.outer(np.cos(u), np.ones_like(v))
    ax3d.plot_surface(xsph, ysph, zsph, color='blue', alpha=0.25, linewidth=0, edgecolor='none')

    # Plot Cape Canaveral as a red "x"
    ax3d.scatter(x0, y0, z0, color='red', marker='x', s=100, label='Cape Canaveral')

    # Plot rocket trajectory by stage (without coastlines)
    colors_map = {
        'stage1': 'orange',
        'stage2': 'blue',
        'coast': 'red',
        'stage2_part2': 'green'
    }
    for stage_name, group_data in df.groupby('stage'):
        ax3d.plot(group_data['x'], group_data['y'], group_data['z'],
                  color=colors_map.get(stage_name, 'black'),
                  label=stage_name)
    ax3d.legend()

    # Set axis labels and limits
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    max_r = df['r'].max()
    margin = 0.1 * max_r
    ax3d.set_xlim([-max_r - margin, max_r + margin])
    ax3d.set_ylim([-max_r - margin, max_r + margin])
    ax3d.set_zlim([-max_r - margin, max_r + margin])

    # Remove extra details for a cleaner look
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.grid(False)
    ax3d._axis3don = False

    # -------------------------------
    # Right Column: Three Vertical Subplots
    # -------------------------------

    # 1) Altitude vs Time (Top row, right column)
    ax1 = fig.add_subplot(gs[0, 1])
    alt = df['r'] - radius_earth
    ax1.plot(df['time'], alt, color='orange')
    ax1.set_ylim(0, 1.0e6)
    ax1.set_xlim(0, df['time'].max())
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Altitude vs Time")
    ax1.grid(True)
    ax1.axhline(y=iss_altitude, linestyle='--', color='gray', label='ISS ~420km')
    ax1.legend()

    # 2) Velocity Components vs Time (Middle row, right column)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(df['time'], df['vel_x'], label='Vx')
    ax2.plot(df['time'], df['vel_y'], label='Vy')
    ax2.plot(df['time'], df['vel_z'], label='Vz')
    ax2.set_xlim(0, df['time'].max())
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity Components vs Time")
    ax2.grid(True)
    ax2.legend()

    # 3) Orbital Velocity vs Time (Bottom row, right column)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(df['time'], df['vel_orb'], color='blue')
    ax3.set_ylabel("Orbital Vel (m/s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim(0, df['time'].max())
    ax3.set_title("Total Velocity vs Time")
    ax3.grid(True)
    ax3.axhline(y=iss_vel_orb, linestyle='--', color='gray', label='ISS Orbital Vel')
    ax3.legend()

    # -------------------------------
    # Encode Figure to Base64 for HTML Embedding
    # -------------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# --------------------------------------------
#   A HELPER TO RUN SIMULATION & RETURN <img>
# --------------------------------------------
def run_simulation(vert_str, burn2_str, coast_str):
    """
    Called from your HTML/JS. Converts string inputs to floats,
    runs the rocket simulation, then returns the final 3-subplot
    figure as a base64-encoded <img> tag.
    """
    v0  = float(vert_str)
    b2  = float(burn2_str)
    cst = float(coast_str)

    df = simulate_gravity_turn_3d(v0, b2, cst)
    img_data = plot_combined_figure(df)

    # Return an HTML <img> with embedded base64 data
    return f"<img src='data:image/png;base64,{img_data}' />"

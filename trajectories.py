import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io, base64

# Set object parameters
C = 0.5                      
x_area = 0.1                  
mass = 100                      

# Create dictionary entries for the planets that will be compared
PLANETS = {
    "Earth": {
        "mu":  3.986004418e14, # gravitational parameter G*M
        "R": 6_371_000, # m -- planet radius
        "rho_0": 1.225, # kg/m^3 -- air density at sea level
        "H": 8400 # scale height for atm. density calculation
    },
    "Mars": {
        "mu": 4.282837e13,
        "R":  3_389_500,
        "rho_0": 0.020,
        "H": 11_100
    },
}

# ------------Functions----------------
def accel(state, planet):
    """
    Takes in the position and velocity
    and computes acceleration components
    that include gravity and drag
    """
    
    # Create state consisting of position and velocity components
    x,y,v_r,v_t = state

    # Get variables of planet being considered
    mu = planet["mu"]
    R = planet["R"]
    rho_0 = planet["rho_0"]
    H_scale = planet["H"]

    # Create vector arrays for position and velocity
    r_vec = np.array([x,y])
    v_vec = np.array([v_r,v_t])

    # Get magnitudes for altitude and speed
    r = np.linalg.norm(r_vec)
    speed = np.linalg.norm(v_vec)

    # Compute gravity vector
    g_vec = (-mu/r**3)*r_vec

    # Compute local atmospheric density (! needs to be updated to use piecewise regimes !)
    h = r - R
    rho_local = rho_0*np.exp(-h / H_scale) if h>= 0 else rho_0 

    # Compute drag vector as opposite of velocity vector 
    drag_vec = ( 
        -0.5 * rho_local * C * x_area/mass * speed * v_vec 
        if speed>0 else np.zeros(2)
    )

    # Create and return acceleration vector from drag and gravity 
    a_vec = g_vec + drag_vec 
    return np.array([v_r, v_t, a_vec[0], a_vec[1]])

def rk4_step(state, dt, planet):
    """
    Uses explicit RK4 integration to iterate
    over timesteps
    """
    
    k1 = accel(state, planet)
    k2 = accel(state + 0.5*dt*k1, planet)
    k3 = accel(state + 0.5*dt*k2, planet)
    k4 = accel(state + dt*k3, planet)

    return state+dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def run_trajectory(v_r0,  planet, *,
                    h0=100_000,
                    dt=0.1, t_max=10000, 
                    store_path=False):
    """
    Function to run the actual trajectory and return:
    -impact time
    -xs,ys arrays 
    """
    # Set initial angles
    theta_prev = 0.0 
    angle_travelled = 0.0

    # Set initial altitude
    r0 = planet["R"] + h0

    # Initialize time and position arrays
    ts, xs, ys = ([], [], []) if store_path else (None, None, None) # record actual trajectory only when necessary

    # Initial state with given horizontal velocity and altitude
    state= np.array([r0, 0.0, 0, v_r0])
    t = 0.0
    last_radius = np.linalg.norm(state[:2])
    min_radius = r0

    # Progress forward until crossing surface or reaching time
    while t< t_max and np.linalg.norm(state[:2]) > planet["R"]:

        # Store path when set to true
        if store_path:
            ts.append(t)
            xs.append(state[0]) 
            ys.append(state[1]) 

        # RK4 Stepper
        state = rk4_step(state, dt, planet)
        t += dt
        
        # Record radius
        last_radius = np.linalg.norm(state[:2])
        
        # Find closest value to starting radius
        min_radius = min(min_radius, last_radius)

        # Calculate angular distance to check to see when a single orbit has been completed
        theta = np.arctan2(state[1], state[0])
        dtheta = theta - theta_prev

        # Unwrap to keep change in theta between -pi and pi
        if dtheta < -np.pi:
            dtheta += 2*np.pi 
        elif dtheta > np.pi:
            dtheta -= 2*np.pi
        angle_travelled += abs(dtheta)
        theta_prev = theta  

        # Break after first revolution of orbit
        if angle_travelled >= 2*math.pi:
            break   

    return (np.array(ts) if store_path else None,
            np.array(xs) if store_path else None, 
            np.array(ys) if store_path else None,
            min_radius)

def find_crit_v(planet,*,
                v_min, v_max, n_points,
                h0, dt, t_max,
                ):
    """
    Finds the critical velocity for a planet, given a velocity range
    and starting altitude. Returns the critical velocity, the tested velocities,
    and their radii of curvature 
    """
    # Starting altitude
    r0 = planet["R"] + h0

    tol_r = 1_000 # meters of acceptable error

    # Create array of velocity values to check 
    v_rs = np.linspace(v_min, v_max, n_points)
    dr_grid = np.empty_like(v_rs)

    # Loop through velocities and calculate radii for each
    for i, v in enumerate(v_rs): 
        *_, min_r = run_trajectory(v, planet, h0=h0, dt=dt, t_max=t_max, store_path=False)

        # Create array of values by subtracting starting radius from closest final radius
        dr_grid[i] = abs(min_r - r0)

    # Raise an error if no value is within tolerance
    if np.all(dr_grid > tol_r):
        raise RuntimeError("No test velocity completed one revolution")

    # Save the velocity that produced the smallest difference in radii
    v_crit = v_rs[np.nanargmin(dr_grid)]

    # Run trajectory for that velocity
    ts, xs, ys, min_r = run_trajectory(v_crit, planet, h0=h0, dt=dt, t_max=t_max,
                                  store_path=True)

    return v_crit, v_rs, dr_grid, ts, xs, ys


def sweep_numeric_vels(planet, alts_km, *,
                       circ_pad=0.15, esc_pad=0.15,
                       n_points=21, dt=1, t_max=5_000):
    """
    Finds critical velocity for range of altitudes using crit V function
    """
    mu = planet["mu"]
    R  = planet["R"]

    v_num_circ = np.empty(len(alts_km))

    # Loop through each altitude and find the critical velocity
    for i, h_km in enumerate(alts_km):
        h_m  = h_km * 1_000
        r0   = R + h_m

        # Compute analytic guesses
        v_circ_ana = np.sqrt(mu / r0)

        # Save critical velocities
        v_num_circ[i], *_ = find_crit_v(
            planet,
            v_min=v_circ_ana,
            v_max=v_circ_ana * (1 + circ_pad),
            n_points=n_points,
            h0=h_m, dt=dt, t_max=t_max
        )

    return v_num_circ

def compare_velocities(planet_name, altitude, vmin, vmax, n_vels):

    """
    Modified version to calculate critical velocity using radius of curvature,
    takes much longer than above version
    """

    # Prompt user for planet and starting altitude
    planet = PLANETS.get(planet_name) 
    altitude= altitude
    altitude_m = altitude*1_000

    v_min = vmin
    v_max = vmax
    n_runs = n_vels

    velocities = np.linspace(v_min, v_max, n_runs)

    paths = [] # create list for paths to be stored
    
    # Loop through velocities and plot each trajectory

    for v in velocities:
        (_, xs, ys, _) = run_trajectory(
            v_r0 = v, 
            planet=planet,
            h0=altitude_m,
            dt=0.1,
            t_max=10_000,
            store_path = True
        )
        paths.append((v,xs,ys))

    # Calculate analytical solutions
    v_circ = np.sqrt(planet["mu"]/(planet["R"] + (altitude_m)))
    v_esc = np.sqrt((2*planet["mu"])/(planet["R"] + altitude_m))

    # Run trajectory for circular velocity
    an_ts, an_xs, an_ys, _ = run_trajectory(
        v_r0 = v_circ, 
        planet=planet, 
        h0 = altitude_m,
        dt=0.1,
        t_max=10_000,
        store_path=True)
    
    # Run trajectory for escape velocity
    escape_ts, escape_xs, escape_ys, _= run_trajectory(
        v_r0 = v_esc, 
        planet=planet, 
        h0 = altitude_m,
        dt=0.1,
        t_max=10_000,
        store_path=True)

    # Plots
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add a filled circle for the planet
    planet_radius_km = planet["R"] / 1e3
    planet_surface = plt.Circle((0, 0), planet_radius_km, color="lightsteelblue", label=f"{planet_name}", zorder=0)
    ax.add_patch(planet_surface)

    # Plot the analytical trajectories    
    ax.plot((escape_xs)/1e3, escape_ys/1e3, label=fr"$v_{{\mathrm{{esc}}}}$={v_esc:.2f} m/s")
    ax.plot((an_xs) / 1e3, an_ys / 1e3, label=fr"$v_{{\mathrm{{ana.}}}}$ = {v_circ:.2f} m/s", color="#CC5500", linestyle="--")
    
    # Plot the tested trajectories
    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                    "tab:olive", "tab:cyan"]
    base_linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5))]
    
    for i, (v, xs, ys) in enumerate(paths):
            color = base_colors[i % len(base_colors)]
            linestyle = base_linestyles[i % len(base_linestyles)]
            ax.plot(xs/1e3, ys/1e3, color=color, linewidth=2, linestyle=linestyle, label=f"{v:.0f} m/s")

    # Limits and labels
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)", fontsize=16)
    ax.set_ylabel("Y (km)", fontsize=16)
    ax.set_ylim((-2*planet["R"])/1e3, (2*planet["R"])/1e3)
    ax.set_xlim((-2*planet["R"])/1e3, (2*planet["R"])/1e3)

    # Axes
    #ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) # force scientific notation
    xt = ax.get_xticks()
    ax.set_xticks(xt[::2]) # set to show every other tick for readability
    yt = ax.get_yticks()
    ax.set_yticks(yt[::2])
    ax.tick_params(axis='both', labelsize=14)

    # Drop legend to two columns automatically if it gets crowded
    ncol = 2 if len(paths) > 10 else 1
    ax.legend(title=f"{int(round(altitude))} km altitude",
              fontsize=9, title_fontsize=11, ncol=ncol)

    ax.grid(True)
    plt.show()

    # Encode figure to Base64 for HTML embedding
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


# ------------- Main function -----------------
def run_simulation(planet, altitude, vmin, vmax, n_vels):
    """
    Called from HTML/JS. Converts string inputs to floats,
    runs the rocket simulation, then returns the final 3-subplot
    figure as a base64-encoded <img> tag.
    """
    planet  = planet
    altitude = float(altitude)
    vmin = float(vmin)
    vmax = float(vmax)
    n_vels = int(n_vels)

    img_data = compare_velocities(planet, altitude, vmin, vmax, n_vels)

    # Return an HTML <img> with embedded base64 data
    return f"<img src='data:image/png;base64,{img_data}' />"
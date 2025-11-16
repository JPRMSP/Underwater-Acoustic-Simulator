# app.py
# Level C — Ultra Advanced Underwater Acoustic Simulator & Beamformer
# Includes: SVP (range-dependent), Snell-based ray tracer (multi-path), Thorp absorption,
# ambient noise mixer, active sonar pulses (LFM/CW/Hyperbolic), matched filter,
# VLA and 2D array beamforming (DAS, MVDR, LCMV), auto-steer, TDOA bearing,
# spectrograms, wavefront animation (Plotly), sonar equation & range prediction,
# rule-based detectors, virtual ocean, and export utilities.
#
# Notes:
# - Designed to be dataset-free and model-free (pure physics + DSP).
# - Use Streamlit for UI. Heavy computations are cached to keep the UI responsive.
# - If running on limited resources, disable "Heavy Physics" toggles or reduce resolution.
#
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, chirp, fftconvolve, correlate
from numpy.linalg import pinv
import plotly.graph_objects as go
import plotly.express as px
from math import sin, cos, radians
import io

st.set_page_config(page_title="Ultra Underwater Acoustic Simulator (Level C)", layout="wide")
st.title("🌊 Ultra Underwater Acoustic Simulator & Beamforming — Level C")

# -----------------------------
# CACHING / UTILITIES
# -----------------------------
@st.cache_data(show_spinner=False)
def linspace_cached(start, stop, num):
    return np.linspace(start, stop, num)

def db(x):
    return 10*np.log10(np.maximum(x, 1e-12))

def save_npy_bytes(arr: np.ndarray):
    b = io.BytesIO()
    np.save(b, arr)
    b.seek(0)
    return b

# -----------------------------
# PHYSICS: SVP (range & depth dependent)
# -----------------------------
@st.cache_data
def munk_profile_depth(depth_m):
    # Munk canonical profile (approx)
    z = depth_m / 1000.0
    return 1500 * (1 + 0.00737 * (z - 1 + np.exp(-z)))

@st.cache_data
def linear_profile_depth(depth_m, grad=0.017):
    return 1480 + grad * depth_m

@st.cache_data
def step_profile_depth(depth_m):
    return np.where(depth_m < 200, 1485, 1520)

@st.cache_data
def build_range_dependent_svp(range_m, depth_m, svp_left, svp_right):
    # Simple linear morph between two SVPs along range (for demonstration)
    # svp_left/right are arrays over depth
    R = len(range_m)
    D = len(depth_m)
    svp = np.zeros((R, D))
    for i, r in enumerate(range_m):
        alpha = i/(R-1)
        svp[i] = (1-alpha)*svp_left + alpha*svp_right
    return svp

# -----------------------------
# THORP & ATTENUATION
# -----------------------------
def thorp_absorption_db_per_km(f_khz):
    f2 = f_khz**2
    # Thorp (approx)
    return 0.11*f2/(1+f2) + 44*f2/(4100+f2) + 2.75e-4*f2 + 0.003

def transmission_loss(r_m, freq_khz, spreading='spherical'):
    alpha = thorp_absorption_db_per_km(freq_khz)
    r_km = r_m/1000.0
    if spreading == 'spherical':
        TL = 20*np.log10(np.maximum(r_m,1.0)) + alpha*r_km
    else:
        TL = 10*np.log10(np.maximum(r_m,1.0)) + alpha*r_km
    return TL

# -----------------------------
# RAY TRACING (Snell's law, range-dependent approx)
# -----------------------------
@st.cache_data
def ray_trace_snell_range_dependent(svp_range_depth, depth_axis, source_depth, launch_angles_deg,
                                    range_step=20.0, max_range=20000.0, surface_reflect=True, bottom_reflect=True):
    """
    svp_range_depth: shape (R, D) speeds in m/s
    depth_axis: array D (m)
    source_depth: single value
    launch_angles_deg: array of angles to launch (deg)
    returns: list of rays each as dict with xs and zs
    """
    R = svp_range_depth.shape[0]
    max_idx = R-1
    range_axis = np.linspace(0, max_range, R)
    dz = depth_axis[1]-depth_axis[0]
    rays = []
    c0 = svp_range_depth[0, np.argmin(np.abs(depth_axis-source_depth))]
    for ang in launch_angles_deg:
        theta = radians(ang)
        x = 0.0
        z = source_depth
        vx = cos(theta)
        vz = sin(theta)
        xs = [x]; zs = [z]
        for i in range(1, R):
            # approximate local sound speed by current range index
            idx = min(i, max_idx)
            c = svp_range_depth[idx, int(np.clip(z/dz, 0, len(depth_axis)-1))]
            # small-step propagation - advance by range_step
            x += vx*range_step
            z += vz*range_step
            # Check reflections
            if surface_reflect and z <= 0:
                z = -z
                vz = -vz
            bottom = depth_axis[-1]
            if bottom_reflect and z >= bottom:
                z = 2*bottom - z
                vz = -vz
            # Snell bending approximate: adjust vx,vz by gradient in c along depth
            # Compute local gradient dc/dz approx by nearest two depth points
            # here we do small heuristic bending proportional to d c/dz
            # find nearest depth index in svp at range idx
            d_idx = int(np.clip(z/dz, 0, len(depth_axis)-2))
            # compute dc/dz using svp at current range index
            local_svp = svp_range_depth[idx]
            dc_dz = (local_svp[d_idx+1] - local_svp[d_idx]) / dz
            # adjust vertical velocity slightly: negative dc/dz -> sound speed decreases with depth -> bend downwards
            bending = -dc_dz * 1e-4  # tuning constant for visualization
            vz += bending
            # re-normalize velocity vector to unit
            norm = np.hypot(vx, vz)
            vx /= norm; vz /= norm
            xs.append(x); zs.append(z)
            if x >= max_range:
                break
        rays.append({'launch_deg': ang, 'xs': np.array(xs), 'zs': np.array(zs)})
    return rays, range_axis

# -----------------------------
# NOISE GENERATORS & MIXER
# -----------------------------
def gen_biological(n, fs):
    t = np.arange(n)/fs
    bursts = np.zeros(n)
    # generate a few pulsed bursts randomly
    for _ in range(np.random.randint(3,10)):
        center = np.random.randint(0, n)
        bw = int(fs*0.005) if fs*0.005>1 else 1
        amp = np.random.uniform(0.2,1.0)
        freq = np.random.uniform(200,1200)
        burst = amp * np.exp(-((np.arange(n)-center)**2)/(2*(bw**2))) * np.sin(2*np.pi*freq*(t))
        bursts += burst
    return bursts

def gen_rain(n, fs):
    # Poisson impulsive events creating broadband energy
    rate = 0.01  # per sample probability
    events = (np.random.rand(n) < rate).astype(float)
    kernel = np.hanning(int(fs*0.01)) if int(fs*0.01)>1 else np.array([1.0])
    rain = np.convolve(events, kernel, mode='same') * np.random.randn(n)
    return rain

def gen_shipping(n, fs):
    t = np.arange(n)/fs
    tones = sum([0.6*np.sin(2*np.pi*f*t + np.random.rand()*2*np.pi) for f in [20,40,60]])
    slow = np.convolve(np.random.randn(n), np.ones(int(0.05*fs))/int(0.05*fs), mode='same')
    return tones + 0.3*slow

def gen_wind(n, fs):
    w = np.random.randn(n)
    return np.convolve(w, np.ones(int(0.02*fs))/int(0.02*fs), mode='same')

def gen_earthquake(n, fs):
    t = np.arange(n)/fs
    chirp_sig = chirp(t, f0=1, f1=50, t1=t[-1], method='linear')
    env = np.exp(-t*2)
    return env * chirp_sig * np.random.uniform(0.5,2.0)

def mix_noise(n, fs, levels):
    noise = np.zeros(n)
    if levels.get('Biological',0)>0:
        noise += levels['Biological']*gen_biological(n, fs)
    if levels.get('Rain',0)>0:
        noise += levels['Rain']*gen_rain(n, fs)
    if levels.get('Shipping',0)>0:
        noise += levels['Shipping']*gen_shipping(n, fs)
    if levels.get('Wind',0)>0:
        noise += levels['Wind']*gen_wind(n, fs)
    if levels.get('Earthquake',0)>0:
        noise += levels['Earthquake']*gen_earthquake(n, fs)
    noise += 0.02*np.random.randn(n)  # low-level broadband background
    return noise

# -----------------------------
# ARRAY SIGNALS, DELAYS & BEAMFORMERS
# -----------------------------
def create_array_signals(angle_deg, freq_hz, fs, n, num_elems=8, spacing=0.75, noise=None):
    t = np.arange(n)/fs
    src = np.sin(2*np.pi*freq_hz*t)
    angle = np.deg2rad(angle_deg)
    c = 1500.0
    delays = spacing*np.arange(num_elems)*np.sin(angle)/c
    arr = np.zeros((num_elems, n))
    for i, d in enumerate(delays):
        shift = int(np.round(d*fs))
        arr[i] = np.roll(src, shift)
    if noise is not None:
        # add independent noise to each hydrophone (correlated via same noise mix)
        for i in range(num_elems):
            arr[i] = arr[i] + noise * (0.5 + 0.5*np.random.rand())
    return arr, src

def delay_and_sum(arr, fs, spacing, steer_deg):
    num_elems, n = arr.shape
    steer = np.deg2rad(steer_deg)
    delays = spacing*np.arange(num_elems)*np.sin(steer)/1500.0
    out = np.zeros(n)
    for i,d in enumerate(delays):
        out += np.roll(arr[i], -int(np.round(d*fs)))
    return out/num_elems

def mvdr(arr, fs, spacing, steer_deg, reg=1e-3):
    num_elems, n = arr.shape
    R = np.cov(arr)
    R = R + reg*np.eye(num_elems)
    steer = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(steer_deg))/1500.0)
    w = pinv(R) @ steer
    denom = (steer.conj().T @ pinv(R) @ steer)
    if denom == 0:
        denom = 1e-6
    w = w / denom
    out = np.real(w.conj().T @ arr)
    return out

def lcmv(arr, fs, spacing, desired_deg, nulls_deg=[]):
    num_elems, n = arr.shape
    steering_des = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(desired_deg))/1500.0)
    A = steering_des.reshape(-1,1)
    for nd in nulls_deg:
        an = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(nd))/1500.0)
        A = np.hstack([A, an.reshape(-1,1)])
    R = np.cov(arr) + 1e-3*np.eye(num_elems)
    Rinv = pinv(R)
    try:
        W = Rinv @ A @ pinv(A.conj().T @ Rinv @ A) @ np.array([1]+[0]*(A.shape[1]-1))
    except:
        # fallback to MVDR-like
        return mvdr(arr, fs, spacing, desired_deg)
    out = np.real(W.conj().T @ arr)
    return out

# TDOA via GCC-PHAT between two sensors
def tdoa_gcc_phat(x, y, fs, max_tau=None):
    n = len(x) + len(y)
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    denom = np.abs(R)
    denom[denom==0] = 1e-8
    R /= denom
    corr = np.fft.irfft(R, n=n)
    corr = np.concatenate((corr[-(len(x)-1):], corr[:len(x)]))
    max_shift = int((max_tau*fs)) if max_tau else len(x)-1
    shift = np.argmax(np.abs(corr)) - max_shift
    tau = shift / float(fs)
    return tau, corr

# Matched filter (pulse compression)
def matched_filter(pulse, data):
    mf = pulse[::-1]
    return fftconvolve(data, mf, mode='same')

# -----------------------------
# DETECTORS (rule-based)
# -----------------------------
def rain_detector(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    bf = np.exp(np.mean(np.log(Pxx+1e-12)))/ (np.mean(Pxx)+1e-12)
    zc = ((sig[:-1]*sig[1:])<0).mean() if len(sig)>1 else 0.0
    score = (1-bf)*0.6 + zc*0.4
    return score > 0.02, score

# -----------------------------
# UI: Controls
# -----------------------------
st.sidebar.header("General Settings")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=4000, step=500)
duration = st.sidebar.number_input("Duration (s)", value=3.0, step=0.5, min_value=0.5)
n = int(fs * duration)

st.sidebar.header("SVP & Ocean")
svp_choice = st.sidebar.selectbox("SVP Type (left/right for range-dep morph)", ["Munk", "Linear", "Step"])
svp_choice_right = st.sidebar.selectbox("SVP Type (right end)", ["Munk", "Linear", "Step"])
max_depth = st.sidebar.slider("Max Depth (m)", 200, 2000, 1000, step=50)
depth_res = st.sidebar.slider("Depth Samples", 100, 1000, 300, step=50)
depth_axis = linspace_cached(0.0, float(max_depth), int(depth_res))

# SVP left/right
if svp_choice == "Munk":
    svp_left = munk_profile_depth(depth_axis)
elif svp_choice == "Linear":
    svp_left = linear_profile_depth(depth_axis)
else:
    svp_left = step_profile_depth(depth_axis)

if svp_choice_right == "Munk":
    svp_right = munk_profile_depth(depth_axis)
elif svp_choice_right == "Linear":
    svp_right = linear_profile_depth(depth_axis)
else:
    svp_right = step_profile_depth(depth_axis)

range_len = st.sidebar.slider("Range Samples for Ray Tracing", 100, 2000, 500, step=50)
range_axis = linspace_cached(0.0, 20000.0, int(range_len))
svp_rd = build_range_dependent_svp(range_axis, depth_axis, svp_left, svp_right)

st.sidebar.header("Noise Mixer (0-1)")
bio_lvl = st.sidebar.slider("Biological", 0.0, 1.0, 0.3)
rain_lvl = st.sidebar.slider("Rain", 0.0, 1.0, 0.2)
ship_lvl = st.sidebar.slider("Shipping", 0.0, 1.0, 0.2)
wind_lvl = st.sidebar.slider("Wind", 0.0, 1.0, 0.1)
eq_lvl = st.sidebar.slider("Earthquake", 0.0, 1.0, 0.0)
noise_levels = {'Biological': bio_lvl, 'Rain': rain_lvl, 'Shipping': ship_lvl, 'Wind': wind_lvl, 'Earthquake': eq_lvl}

st.sidebar.header("Array & Source")
num_elements = st.sidebar.slider("Array Elements (linear)", 2, 16, 8)
spacing = st.sidebar.slider("Element Spacing (m)", 0.2, 1.5, 0.75, step=0.05)
src_freq = st.sidebar.slider("Source Frequency (Hz)", 20, 2000, 300)
src_angle = st.sidebar.slider("Source Angle (deg)", -60, 60, 15)
source_depth = st.sidebar.slider("Source Depth (m)", 0, int(max_depth), int(max_depth/5))

st.sidebar.header("Active Sonar")
use_active = st.sidebar.checkbox("Use Active Sonar Pulse", value=True)
pulse_type = st.sidebar.selectbox("Pulse Type", ["LFM", "CW", "Hyperbolic"])
pulse_bw = st.sidebar.slider("Pulse Bandwidth (Hz)", 50, 2000, 400)
pulse_len_ms = st.sidebar.slider("Pulse Length (ms)", 10, 200, 80)

st.sidebar.header("Beamforming & Algorithms")
bf_method = st.sidebar.selectbox("Beamforming Method", ["Delay-and-Sum", "MVDR", "LCMV"])
steer_angle = st.sidebar.slider("Manual Steering Angle (deg)", -90, 90, 0)
auto_steer = st.sidebar.checkbox("Enable Auto-Steer (Max Energy)", value=True)
nulls_text = st.sidebar.text_input("LCMV Null Angles (comma separated)", value="-30,30")
nulls_list = []
try:
    nulls_list = [float(x.strip()) for x in nulls_text.split(",") if x.strip()!='']
except:
    nulls_list = []

st.sidebar.header("Sonar Equation")
SL = st.sidebar.number_input("Source Level SL (dB re 1µPa @1m)", value=190.0)
TS = st.sidebar.number_input("Target Strength TS (dB)", value=10.0)
NL = st.sidebar.number_input("Noise Level NL (dB re 1µPa/Hz)", value=60.0)
DI = st.sidebar.number_input("Directivity Index DI (dB)", value=10.0)
spreading = st.sidebar.selectbox("Spreading Type", ["spherical", "cylindrical"])
freq_for_abs_khz = st.sidebar.slider("Frequency for Absorption (kHz)", 0.1, 50.0, 0.5)

st.sidebar.header("Performance / Physics toggles")
heavy_physics = st.sidebar.checkbox("Enable High-Resolution Ray Tracing & Wavefront Animation", value=True)
animation_steps = st.sidebar.slider("Wavefront animation frames", 10, 80, 24)
show_wavefront = st.sidebar.checkbox("Show Wavefront Animation", value=True)

# -----------------------------
# SIGNALS & NOISE
# -----------------------------
noise_mix = mix_noise(n, fs, noise_levels)  # ensure defined before use (fixes prior NameError)

# Create array signals (passive + noise)
arr_passive, src_signal = create_array_signals(src_angle, src_freq, fs, n,
                                               num_elems=num_elements, spacing=spacing, noise=noise_mix)

# Active pulse generation & injection (multipath simplified via delayed arrivals)
if use_active:
    t_p = np.arange(int(fs * (pulse_len_ms/1000.0))) / fs
    if pulse_type == "LFM":
        pulse = chirp(t_p, f0=max(1, src_freq-pulse_bw/2), f1=src_freq+pulse_bw/2, t1=t_p[-1], method='linear')
    elif pulse_type == "CW":
        pulse = np.sin(2*np.pi*src_freq*t_p)
    else:
        pulse = chirp(t_p, f0=max(1, src_freq-pulse_bw/2), f1=src_freq+pulse_bw/2, t1=t_p[-1], method='quadratic')
    # build active array signals by applying delays consistent with geometry & multipath (direct + surface + bottom)
    arr_active = np.copy(arr_passive)
    # direct path injection
    angle_rad = np.deg2rad(src_angle)
    c = 1500.0
    delays = spacing*np.arange(num_elements)*np.sin(angle_rad)/c
    for i, d in enumerate(delays):
        shift = int(np.round(d*fs))
        arr_active[i, :len(pulse)] += np.roll(pulse, shift) * 1.0
    # simple multipath: surface reflection with inverted polarity and extra delay
    for i, d in enumerate(delays):
        shift_extra = int(np.round((2*source_depth/c + d)*fs))
        arr_active[i, :len(pulse)] += np.roll(-pulse*0.6, shift_extra)
    # bottom reflection: assuming bottom at max_depth
    bottom_delay = int(np.round((2*(max_depth-source_depth)/c)*fs))
    for i in range(num_elements):
        arr_active[i, :len(pulse)] += np.roll(pulse*0.4, bottom_delay)
else:
    arr_active = arr_passive
    pulse = None

avg_hydro = arr_active.mean(axis=0)

# -----------------------------
# BEAMFORMING & AUTO-STEER
# -----------------------------
def compute_beamformed_output(arr, method, steer, nulls=[]):
    if method == "Delay-and-Sum":
        return delay_and_sum(arr, fs, spacing, steer)
    elif method == "MVDR":
        return mvdr(arr, fs, spacing, steer)
    else:
        return lcmv(arr, fs, spacing, steer, nulls)

if auto_steer:
    # search coarse angles
    scan_angles = np.linspace(-90, 90, 181)
    energies = []
    for a in scan_angles:
        y = delay_and_sum(arr_active, fs, spacing, a)
        energies.append(np.mean(y**2))
    best_angle = float(scan_angles[np.argmax(energies)])
else:
    best_angle = steer_angle
    scan_angles = None
    energies = None

bf_steer = best_angle if auto_steer else steer_angle
bf_out = compute_beamformed_output(arr_active, bf_method, bf_steer, nulls_list)

# TDOA bearing estimate between first two sensors
try:
    tau, corr = tdoa_gcc_phat(arr_active[0], arr_active[1], fs, max_tau=spacing/1500.0)
    tdoa_angle = np.degrees(np.arcsin(np.clip(1500.0 * tau / spacing, -0.999, 0.999)))
except Exception as e:
    tdoa_angle = np.nan

# Matched filter (active)
if use_active and pulse is not None:
    matched = matched_filter(pulse, avg_hydro)
else:
    matched = None

# Spectrograms
f_bf, t_bf, Sxx_bf = spectrogram(bf_out, fs=fs, nperseg=min(512, n//4))
f_avg, t_avg, Sxx_avg = spectrogram(avg_hydro, fs=fs, nperseg=min(512, n//4))

# Rain detector
rain_flag, rain_score = rain_detector(avg_hydro, fs)

# -----------------------------
# RAY TRACING & WAVEFRONT (heavy)
# -----------------------------
launch_angles = np.linspace(-60, 60, 31)
if heavy_physics:
    rays, ray_range_axis = ray_trace_snell_range_dependent(svp_rd, depth_axis, source_depth, launch_angles,
                                                           range_step=range_axis[1]-range_axis[0],
                                                           max_range=range_axis[-1],
                                                           surface_reflect=True, bottom_reflect=True)
else:
    rays = []
    ray_range_axis = range_axis

# Wavefront animation (approx): expand circular wavefronts from source depth projected in (range, depth)
def build_wavefront_frames(num_frames=24, max_r=20000, source_pos=(0, source_depth), num_points=180):
    frames = []
    rs = np.linspace(0, max_r, num_frames+1)[1:]
    theta = np.linspace(0, 2*np.pi, num_points)
    for r in rs:
        xs = source_pos[0] + r*np.cos(theta)
        zs = source_pos[1] + r*np.sin(theta)  # not physically accurate in depth axis, but illustrative
        frames.append((xs, zs))
    return frames

if show_wavefront and heavy_physics:
    frames = build_wavefront_frames(animation_steps, max_r=range_axis[-1], source_pos=(0, source_depth))
else:
    frames = []

# -----------------------------
# SONAR EQUATION & RANGE PREDICTION
# -----------------------------
alpha_km = thorp_absorption_db_per_km(freq_for_abs_khz)
ranges_m = np.linspace(10, 50000, 500)
TLs = transmission_loss(ranges_m, freq_for_abs_khz, spreading=spreading)
SNRs_passive = SL - TLs + DI - NL
max_det_idx = np.where(SNRs_passive > 0)[0]
max_detect_range = float(ranges_m[max_det_idx].max()) if max_det_idx.size>0 else None

# -----------------------------
# VISUALIZATION LAYOUT
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Sound Velocity Profile (Range-dependent overview)")
    # show left and right SVP plus a few intermediate ranges
    fig_svp, ax_svp = plt.subplots(figsize=(4,4))
    ax_svp.plot(svp_left, depth_axis, label='SVP Left')
    ax_svp.plot(svp_right, depth_axis, label='SVP Right')
    mid_idx = svp_rd.shape[0]//2
    ax_svp.plot(svp_rd[mid_idx], depth_axis, label='SVP Mid (range)')
    ax_svp.invert_yaxis()
    ax_svp.set_xlabel("Sound Speed (m/s)"); ax_svp.set_ylabel("Depth (m)")
    ax_svp.legend()
    st.pyplot(fig_svp)

    st.subheader("Ray Fan (Snell Range-Dependent, multi-path)")
    fig_rt, ax_rt = plt.subplots(figsize=(6,3))
    for r in rays:
        ax_rt.plot(r['xs'], r['zs'], linewidth=0.8)
    ax_rt.set_xlim(0, range_axis[-1])
    ax_rt.set_ylim(depth_axis[-1], 0)
    ax_rt.set_xlabel("Range (m)"); ax_rt.set_ylabel("Depth (m)")
    st.pyplot(fig_rt)

    st.subheader("Ambient Noise PSD (avg hydrophone)")
    f_noise, Pxx_noise = welch(avg_hydro, fs=fs, nperseg=min(1024, n))
    fig_n, ax_n = plt.subplots()
    ax_n.semilogy(f_noise, Pxx_noise + 1e-12)
    ax_n.set_xlabel("Frequency (Hz)"); ax_n.set_ylabel("PSD")
    st.pyplot(fig_n)

    st.subheader("Sonar Equation — Range Prediction (Passive)")
    st.write(f"Absorption @ {freq_for_abs_khz} kHz = {alpha_km:.4f} dB/km")
    st.line_chart({"Range (m)": ranges_m, "Passive SNR (dB)": SNRs_passive})
    if max_detect_range:
        st.success(f"Estimated max detection range (Passive, SNR>0): **{max_detect_range:.0f} m**")
    else:
        st.warning("No detection (Passive SNR <= 0) within 50 km using current parameters.")

with col2:
    st.subheader("Array Signals & Beamforming Outputs")
    fig_arr, ax_arr = plt.subplots(figsize=(6,3))
    t = np.arange(n)/fs
    # plot first 4 hydrophones
    for i in range(min(4, arr_active.shape[0])):
        ax_arr.plot(t[:min(len(t),int(fs*0.5))], arr_active[i,:min(len(t),int(fs*0.5))] + i*2, label=f"El {i+1}")
    ax_arr.set_xlabel("Time (s)")
    st.pyplot(fig_arr)

    st.subheader("Beamformed Output (waveform + spectrogram)")
    fig_bf, ax_bf = plt.subplots(figsize=(6,2))
    ax_bf.plot(t[:min(len(t),int(fs*0.5))], bf_out[:min(len(t),int(fs*0.5))])
    ax_bf.set_xlabel("Time (s)")
    st.pyplot(fig_bf)

    fig_sp, ax_sp = plt.subplots(figsize=(6,3))
    ax_sp.pcolormesh(t_bf, f_bf, 10*np.log10(Sxx_bf+1e-12), shading='gouraud')
    ax_sp.set_ylabel("Freq (Hz)"); ax_sp.set_xlabel("Time (s)")
    st.pyplot(fig_sp)

st.markdown("### Beamforming Summary")
st.write(f"Method: **{bf_method}** — Steering used: **{bf_steer:.1f}°** (Auto-steer {'ON' if auto_steer else 'OFF'})")
st.write(f"TDOA bearing estimate (elems 1-2): **{tdoa_angle:.2f}°**")
st.write(f"Rain Detector: **{'Rain Detected' if rain_flag else 'No Rain'}** (score: {rain_score:.4f})")
if auto_steer and scan_angles is not None:
    fig_scan, ax_scan = plt.subplots()
    ax_scan.plot(scan_angles, energies)
    ax_scan.set_xlabel("Angle (deg)"); ax_scan.set_ylabel("Energy")
    st.pyplot(fig_scan)

# Matched filter result
if matched is not None:
    st.subheader("Matched Filter Output (Active Sonar)")
    fig_mf, ax_mf = plt.subplots()
    ax_mf.plot(np.arange(len(matched))/fs, matched)
    ax_mf.set_xlabel("Time (s)")
    st.pyplot(fig_mf)

# 2D Beam Pattern (simple HxV grid)
st.subheader("2D Beam Pattern (Array Grid Approximation)")
Nx = st.sidebar.slider("Array Nx (for 2D pattern visualization)", 1, 8, 4)
Ny = st.sidebar.slider("Array Ny (for 2D pattern visualization)", 1, 8, 4)
pattern_freq = st.sidebar.slider("Pattern Frequency (Hz)", 100, 2000, 500)
# compute rudimentary 2D pattern by summing element phase responses
az = np.linspace(-90, 90, 181)
el = np.linspace(-90, 90, 181)
AZ, EL = np.meshgrid(az, el)
k = 2*np.pi*(pattern_freq)/1500.0
xs = (np.arange(Nx) - (Nx-1)/2) * spacing
ys = (np.arange(Ny) - (Ny-1)/2) * spacing
P = np.zeros_like(AZ, dtype=float)
for xi in xs:
    for yj in ys:
        phase = np.exp(-1j * k * (xi*np.sin(np.deg2rad(AZ))*np.cos(np.deg2rad(EL)) + yj*np.sin(np.deg2rad(EL))))
        P += np.abs(phase)**2
PdB = 10*np.log10(np.maximum(P/P.max(), 1e-6))
fig_heat = go.Figure(data=go.Heatmap(z=PdB, x=az, y=el, colorbar=dict(title="dB")))
fig_heat.update_layout(xaxis_title="Azimuth (deg)", yaxis_title="Elevation (deg)", height=450)
st.plotly_chart(fig_heat, use_container_width=True)

# Wavefront animation (Plotly scatter frames)
if show_wavefront and len(frames)>0:
    st.subheader("Wavefront Animation (Illustrative)")
    fig_w = go.Figure(
        data=[go.Scatter(x=frames[0][0], y=frames[0][1], mode='markers', marker=dict(size=2))],
        layout=go.Layout(
            xaxis=dict(range=[0, range_axis[-1]*1.05], autorange=False),
            yaxis=dict(range=[depth_axis[-1], 0], autorange=False),
            title="Expanding Wavefronts (illustration)",
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 200, "redraw": True},
                                                         "fromcurrent": True, "transition": {"duration": 0}}])])]))
    frames_plot = [go.Frame(data=[go.Scatter(x=fr[0], y=fr[1], mode='markers', marker=dict(size=2))]) for fr in frames]
    fig_w.frames = frames_plot
    st.plotly_chart(fig_w, use_container_width=True)
else:
    st.info("Wavefront animation disabled or heavy_physics off.")

# Virtual Ocean simple map
st.subheader("Virtual Ocean — Place & Compare Sources")
colA, colB, colC = st.columns(3)
with colA:
    vx_src_range = st.number_input("Virtual Source Range (m)", min_value=10, max_value=int(range_axis[-1]), value=2000)
with colB:
    vx_src_depth = st.number_input("Virtual Source Depth (m)", min_value=0, max_value=int(max_depth), value=source_depth)
with colC:
    vx_strength = st.slider("Relative Strength", 0.1, 5.0, 1.0)
fig_vo, ax_vo = plt.subplots(figsize=(6,3))
ax_vo.scatter([0, vx_src_range], [100, vx_src_depth], c=['blue','red'])
ax_vo.annotate("Array", (0,100)); ax_vo.annotate("Source", (vx_src_range, vx_src_depth))
ax_vo.set_xlim(0, max(2000, vx_src_range*1.2)); ax_vo.set_ylim(depth_axis[-1], 0)
ax_vo.set_xlabel("Range (m)"); ax_vo.set_ylabel("Depth (m)")
st.pyplot(fig_vo)

# Export capabilities
st.subheader("Export / Download")
if st.button("Download Beamformed Output (.npy)"):
    b = save_npy_bytes(bf_out.astype(np.float32))
    st.download_button("Download .npy", b, file_name="beamformed_output.npy")

if st.button("Download Average Hydrophone (.npy)"):
    b2 = save_npy_bytes(avg_hydro.astype(np.float32))
    st.download_button("Download .npy", b2, file_name="avg_hydrophone.npy")

# Short help mapping to FI9006 syllabus
st.markdown("---")
st.markdown("### Mapping to FI9006 syllabus")
st.write("""
- **UNIT I:** SVP, Snell (ray tracer), multi-path, shallow/deep water propagation.  
- **UNIT II:** Ambient noise sources, variability, PSD and spectrogram analysis.  
- **UNIT III:** Signals, matched filtering (pulse compression), spectrograms, filters.  
- **UNIT IV:** Sonar equation, transducer array modelling, DAS/MVDR/LCMV beamforming, TDOA bearing.  
- **UNIT V:** DSP heavy processing implemented; optimized for Python.  
""")

st.markdown("### Notes / Troubleshooting")
st.write("""
- If Streamlit becomes unresponsive, reduce `Duration`, `Depth Samples`, `Range Samples`, or disable `High-Resolution Ray Tracing`.  
- For very large `array elements` or `animation frames`, runtime cost increases significantly.  
- This app is intentionally modular — I can split this into files, add a Colab launcher, or make a GitHub repo with CI on demand.
""")

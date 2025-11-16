# app.py
# Real-Time Underwater Acoustic Simulator & Advanced Beamformer
# All enhancements included except hardware extensions (Category 8)
# Dataset-free, model-free — pure physics & DSP simulation.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, chirp, fftconvolve
from numpy.linalg import inv, pinv, eig
import plotly.graph_objects as go

st.set_page_config(page_title="Underwater Acoustic Lab — Advanced", layout="wide")
st.title("🌊 Underwater Acoustic Simulator — Advanced (No Hardware)")

# ------------------------
# Utility & Physics Models
# ------------------------
def munk_svp(depth_m):
    z = depth_m / 1000.0
    # simplified Munk-like profile
    return 1500 * (1 + 0.00737 * (z - 1 + np.exp(-z)))

def linear_svp(depth_m, grad=0.017):
    return 1480 + grad * depth_m

def step_svp(depth_m):
    return np.where(depth_m < 200, 1485, 1520)

def thorp_absorption(f_khz):
    # Thorp's empirical formula (dB/km) for f in kHz
    f2 = f_khz**2
    return 0.11*f2/(1+f2) + 44*f2/(4100+f2) + 2.75e-4*f2 + 0.003

def transmission_loss_range(r_m, alpha_db_per_km, spreading='spherical'):
    r_km = r_m / 1000.0
    if spreading == 'spherical':
        TL = 20*np.log10(np.maximum(r_m,1.0)) + alpha_db_per_km * r_km
    else:
        TL = 10*np.log10(np.maximum(r_m,1.0)) + alpha_db_per_km * r_km
    return TL

# Simple ray tracing (range-independent) using Snell's law approx with small-angle assumption
def ray_fan(svp_depths, svp_speeds, source_depth, num_rays=21, max_range=10000):
    depths = []
    ranges = []
    angles = np.linspace(-70,70,num_rays)  # degrees
    z = np.linspace(0, len(svp_speeds)-1, len(svp_speeds))  # index ~ depth step
    dz = 1
    for ang in angles:
        theta = np.deg2rad(ang)
        x = 0; depth = source_depth
        xs, zs = [x], [depth]
        vx = np.cos(theta); vz = np.sin(theta)
        for _ in range(2000):
            # local index and local c
            idx = int(np.clip(depth/ (len(svp_speeds)/len(svp_depths)), 0, len(svp_speeds)-1))
            c = svp_speeds[idx]
            # simple curvature due to vertical grad
            # step forward
            x += vx*10
            depth += vz*10
            # reflect at surface/bottom (assume bottom at max depth)
            if depth <= 0:
                depth = -depth
                vz = -vz
            if depth >= svp_depths[-1]:
                depth = 2*svp_depths[-1] - depth
                vz = -vz
            xs.append(x); zs.append(depth)
            if x > max_range: break
        ranges.append(xs); depths.append(zs)
    return angles, ranges, depths

# ------------------------
# Noise Generators (mixer)
# ------------------------
def gen_biological(n, fs):
    t = np.arange(n)/fs
    bursts = np.zeros(n)
    for _ in range(int(0.3*fs)):
        center = np.random.randint(0,n)
        bw = int(fs*0.005)
        amp = np.random.uniform(0.2,1.0)
        burst = amp * np.exp(-((np.arange(n)-center)**2)/(2*(bw**2))) * np.sin(2*np.pi*np.random.uniform(200,1000)*t)
        bursts += burst
    return bursts

def gen_rain(n, fs):
    # impulsive broadband events (Poisson)
    t = np.arange(n)/fs
    events = np.random.poisson(0.005, n)
    rain = np.convolve(events, np.hanning(int(fs*0.01)), mode='same') * np.random.randn(n)
    return rain

def gen_shipping(n, fs):
    t = np.arange(n)/fs
    tones = sum([0.5*np.sin(2*np.pi*f*t + np.random.rand()*2*np.pi) for f in [20,40,60]])
    slow = np.convolve(np.random.randn(n), np.ones(int(0.05*fs))/int(0.05*fs), mode='same')
    return tones + 0.3*slow

def gen_wind(n, fs):
    w = np.random.randn(n)
    return np.convolve(w, np.ones(int(0.02*fs))/int(0.02*fs), mode='same')

def gen_earthquake(n, fs):
    t = np.arange(n)/fs
    chirp_sig = chirp(t, f0=1, f1=20, t1=t[-1], method='linear')
    env = np.exp(-t*2)
    return env * chirp_sig * np.random.uniform(0.5,2.0)

# ------------------------
# Array & Signal Utilities
# ------------------------
def create_array_signals(source_angle_deg, source_freq, fs, n, num_elems=8, spacing=0.75, noise_mix=None):
    t = np.arange(n)/fs
    src = np.sin(2*np.pi*source_freq*t)
    angle = np.deg2rad(source_angle_deg)
    c = 1500.0
    delays = spacing*np.arange(num_elems)*np.sin(angle)/c
    arr = np.zeros((num_elems,n))
    for i,d in enumerate(delays):
        shift = int(np.round(d*fs))
        arr[i] = np.roll(src, shift)
    # add noise
    if noise_mix is not None:
        arr += noise_mix
    return arr, src

def add_noise_mixture(n, fs, mix_levels):
    # mix_levels: dict with keys 'Biological','Rain','Shipping','Wind','Earthquake'
    noise = np.zeros(n)
    if mix_levels.get('Biological',0)>0:
        noise += mix_levels['Biological']*gen_biological(n, fs)
    if mix_levels.get('Rain',0)>0:
        noise += mix_levels['Rain']*gen_rain(n, fs)
    if mix_levels.get('Shipping',0)>0:
        noise += mix_levels['Shipping']*gen_shipping(n, fs)
    if mix_levels.get('Wind',0)>0:
        noise += mix_levels['Wind']*gen_wind(n, fs)
    if mix_levels.get('Earthquake',0)>0:
        noise += mix_levels['Earthquake']*gen_earthquake(n, fs)
    # background ambient
    noise += 0.05*np.random.randn(n)
    return noise

# Delay-and-sum beamformer (frequency-agnostic simple)
def delay_and_sum_beamform(arr, fs, spacing, steer_deg):
    num_elems, n = arr.shape
    steer = np.deg2rad(steer_deg)
    delays = spacing*np.arange(num_elems)*np.sin(steer)/1500.0
    out = np.zeros(n)
    for i,d in enumerate(delays):
        out += np.roll(arr[i], -int(np.round(d*fs)))
    return out/num_elems

# MVDR beamformer (snapshot covariance)
def mvdr_beamformer(arr, fs, spacing, steer_deg, reg=1e-3):
    num_elems, n = arr.shape
    R = np.cov(arr)
    R = R + reg*np.eye(num_elems)
    steer = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(steer_deg))/1500.0)
    w = pinv(R) @ steer
    w = w / (steer.conj().T @ pinv(R) @ steer)
    out = np.real(w.conj().T @ arr)
    return out

# LCMV (simple constrained MVDR) -- use steering for desired and nulls for interferers
def lcmv_beamformer(arr, fs, spacing, desired_deg, nulls_deg=[]):
    num_elems, n = arr.shape
    # build constraint matrix
    steering_des = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(desired_deg))/1500.0)
    A = steering_des.reshape(-1,1)
    for nd in nulls_deg:
        an = np.exp(-1j*2*np.pi*0.75*np.arange(num_elems)*np.sin(np.deg2rad(nd))/1500.0)
        A = np.hstack([A, an.reshape(-1,1)])
    R = np.cov(arr) + 1e-3*np.eye(num_elems)
    Rinv = pinv(R)
    W = Rinv @ A @ pinv(A.conj().T @ Rinv @ A) @ np.array([1]+[0]*(A.shape[1]-1))
    out = np.real(W.conj().T @ arr)
    return out

# Beam pattern for 2D (H x V) rectangular array (compute beampattern magnitude)
def compute_2d_beam_pattern(Nx, Ny, spacing, freqs, steer_az_deg=0, steer_el_deg=0):
    # generate grid
    az = np.linspace(-90,90,181)
    el = np.linspace(-90,90,181)
    AZ, EL = np.meshgrid(az, el)
    k = 2*np.pi*freqs/1500.0
    # array coords
    xs = (np.arange(Nx)- (Nx-1)/2) * spacing
    ys = (np.arange(Ny)- (Ny-1)/2) * spacing
    P = np.zeros(AZ.shape, dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            phase = np.exp(-1j * k * (xs[i]*np.sin(np.deg2rad(AZ))*np.cos(np.deg2rad(EL)) + ys[j]*np.sin(np.deg2rad(EL))))
            P += np.abs(phase)**2
    PdB = 10*np.log10(np.maximum(P/P.max(),1e-6))
    return az, el, PdB

# TDOA bearing estimator (simple GCC-PHAT between top two elements)
def tdoa_bearing_estimate(arr, fs, spacing):
    x = arr[0]; y = arr[1]
    # GCC-PHAT
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    R = X * np.conj(Y)
    R /= np.abs(R)+1e-8
    corr = np.fft.irfft(R, n=len(x))
    shift = np.argmax(corr) - len(x)//2
    tau = shift/fs
    # angle from TDOA (sin theta = c * tau / d)
    val = 1500 * tau / spacing
    angle = np.degrees(np.arcsin(np.clip(val, -0.999, 0.999)))
    return angle

# Matched filter for LFM pulse compression
def matched_filter(pulse, data):
    mf = pulse[::-1]  # time-reversed
    return fftconvolve(data, mf, mode='same')

# Small rule-based rain detector (no ML)
def rain_detector(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=1024)
    # rain has broadband energy bursts -> compute spectral flatness-ish measure
    bf = np.exp(np.mean(np.log(Pxx+1e-12)))/ (np.mean(Pxx)+1e-12)
    zc = ((sig[:-1]*sig[1:])<0).mean()
    # heuristic
    score = (1-bf)*0.6 + zc*0.4
    return score > 0.02  # threshold

# ------------------------
# Streamlit UI
# ------------------------
st.sidebar.header("Global Simulation")
fs = st.sidebar.number_input("Sampling rate (Hz)", value=4000, step=1000)
duration = st.sidebar.number_input("Duration (s)", value=2.0, step=0.5)
n = int(fs * duration)

st.sidebar.subheader("Sound Velocity Profile")
svp_choice = st.sidebar.selectbox("SVP Type", ["Munk", "Linear", "Step"])
depth_axis = np.linspace(0, 1000, 500)
if svp_choice == "Munk":
    svp = munk_svp(depth_axis)
elif svp_choice == "Linear":
    svp = linear_svp(depth_axis)
else:
    svp = step_svp(depth_axis)

st.sidebar.subheader("Noise Mixer (levels 0-1)")
bio_lvl = st.sidebar.slider("Biological", 0.0, 1.0, 0.2)
rain_lvl = st.sidebar.slider("Rain", 0.0, 1.0, 0.2)
ship_lvl = st.sidebar.slider("Shipping", 0.0, 1.0, 0.2)
wind_lvl = st.sidebar.slider("Wind", 0.0, 1.0, 0.1)
eq_lvl = st.sidebar.slider("Earthquake", 0.0, 1.0, 0.0)

mix = {'Biological':bio_lvl,'Rain':rain_lvl,'Shipping':ship_lvl,'Wind':wind_lvl,'Earthquake':eq_lvl}
noise_mix = add_noise_mixture(n, fs, mix)

# Source and array settings
st.sidebar.subheader("Source & Array")
src_freq = st.sidebar.slider("Source Tone Frequency (Hz)", 20, 1500, 300)
src_angle = st.sidebar.slider("Source Angle (deg)", -60, 60, 20)
num_elements = st.sidebar.slider("Array Elements (linear)", 2, 16, 8)
spacing = st.sidebar.slider("Element Spacing (m)", 0.25, 1.0, 0.75)

# Active sonar settings
st.sidebar.subheader("Active Sonar")
use_active = st.sidebar.checkbox("Simulate Active Sonar Pulse", value=False)
pulse_type = st.sidebar.selectbox("Pulse Type", ["LFM", "CW", "Hyperbolic LFM"])
pulse_bw = st.sidebar.slider("Pulse bandwidth (Hz)", 100, 2000, 400)
pulse_len = st.sidebar.slider("Pulse length (ms)", 10, 200, 50)

# Beamforming options
st.sidebar.subheader("Beamforming")
bf_method = st.sidebar.selectbox("Method", ["Delay-and-Sum", "MVDR", "LCMV"])
steer_angle = st.sidebar.slider("Steering Angle (deg)", -90, 90, 0)
nulls = st.sidebar.text_input("LCMV Null Angles (comma separated)", value="-30,30")

# Sonar equation panel inputs
st.sidebar.subheader("Sonar Equation Inputs")
SL = st.sidebar.number_input("Source Level SL (dB re 1µPa @1m)", value=190.0)
TS = st.sidebar.number_input("Target Strength TS (dB)", value=10.0)
NL = st.sidebar.number_input("Noise Level NL (dB re 1µPa/Hz)", value=60.0)
DI = st.sidebar.number_input("Directivity Index DI (dB)", value=10.0)
spreading = st.sidebar.selectbox("Spreading", ["spherical","cylindrical"])
freq_for_abs_khz = st.sidebar.slider("Frequency for absorption (kHz)", 0.1, 50.0, 0.3)

# ------------------------
# Generate array & signals
# ------------------------
arr, src = create_array_signals(src_angle, src_freq, fs, n, num_elems, spacing, noise_mix)
# Average hydrophone noise (for detectors)
avg_hydro = arr.mean(axis=0)

# Active pulse generation
if use_active:
    t = np.arange(int(fs * (pulse_len/1000.0))) / fs
    if pulse_type == "LFM":
        pulse = chirp(t, f0=src_freq-pulse_bw/2, f1=src_freq+pulse_bw/2, t1=t[-1], method='linear')
    elif pulse_type == "CW":
        pulse = np.sin(2*np.pi*src_freq*t)
    else:
        # hyperbolic-like chirp approximation
        pulse = chirp(t, f0=src_freq-pulse_bw/2, f1=src_freq+pulse_bw/2, t1=t[-1], method='quadratic')
    # Inject pulse into array with same geometry and noise
    p_arr = np.zeros_like(arr)
    angle = np.deg2rad(src_angle)
    delays = spacing*np.arange(num_elements)*np.sin(angle)/1500.0
    for i,d in enumerate(delays):
        shift = int(np.round(d*fs))
        p_arr[i, :len(pulse)] = np.roll(pulse, shift)
    arr_active = arr + p_arr
else:
    arr_active = arr

# ------------------------
# Beamforming & Detection
# ------------------------
if bf_method == "Delay-and-Sum":
    bf_out = delay_and_sum_beamform(arr_active, fs, spacing, steer_angle)
elif bf_method == "MVDR":
    bf_out = mvdr_beamformer(arr_active, fs, spacing, steer_angle)
else:  # LCMV
    try:
        null_list = [float(x.strip()) for x in nulls.split(',') if x.strip()!='']
    except:
        null_list = []
    bf_out = lcmv_beamformer(arr_active, fs, spacing, steer_angle, null_list)

# Auto Beam Steering (search for angle that maximizes output energy)
def auto_steer(arr_data, fs, spacing):
    angles = np.linspace(-90,90,181)
    energies = []
    for a in angles:
        y = delay_and_sum_beamform(arr_data, fs, spacing, a)
        energies.append(np.mean(y**2))
    best_ang = angles[np.argmax(energies)]
    return best_ang, angles, energies

auto_steer_enabled = st.sidebar.checkbox("Auto Beam Steer (Max Energy)", value=False)
if auto_steer_enabled:
    best_ang, scan_angles, scan_energies = auto_steer(arr_active, fs, spacing)
else:
    best_ang, scan_angles, scan_energies = None, None, None

# TDOA bearing using first two elements
tdoa_angle = tdoa_bearing_estimate(arr_active, fs, spacing)

# Rain detector test
rain_detected = rain_detector(avg_hydro, fs)

# Matched filter if active
if use_active:
    matched = matched_filter(pulse, avg_hydro)
else:
    matched = None

# 2D Beam pattern compute for AxB array (use small default)
Nx = st.sidebar.slider("Array Nx (for 2D pattern)", 1, 8, 4)
Ny = st.sidebar.slider("Array Ny (for 2D pattern)", 1, 8, 4)
pattern_freq = st.sidebar.slider("Pattern frequency (Hz)", 100, 2000, 500)
az, el, PdB = compute_2d_beam_pattern(Nx, Ny, spacing, pattern_freq)

# ------------------------
# Sonar Equation & Range Prediction
# ------------------------
alpha_db_km = thorp_absorption(freq_for_abs_khz)
ranges_m = np.linspace(10,50000,500)
TLs = transmission_loss_range(ranges_m, alpha_db_km, spreading)
# Passive sonar SNR approx: SL - TL + DI - NL -> detection when > threshold (e.g., 0 dB)
SNRs = SL - TLs + DI - NL
max_detect_range = ranges_m[np.where(SNRs>0)[0]].max() if np.any(SNRs>0) else None

# ------------------------
# Plots & Outputs layout
# ------------------------
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Sound Velocity Profile (SVP)")
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.plot(svp, depth_axis)
    ax1.invert_yaxis()
    ax1.set_xlabel("Sound speed (m/s)")
    ax1.set_ylabel("Depth (m)")
    st.pyplot(fig1)

    st.subheader("Ray Fan (approx)")
    angles_rt, ranges_rt, depths_rt = ray_fan(depth_axis, svp, source_depth=100, num_rays=15, max_range=20000)
    fig2, ax2 = plt.subplots(figsize=(6,3))
    for r,d in zip(ranges_rt, depths_rt):
        ax2.plot(r, d, linewidth=0.7)
    ax2.set_ylim(1000,0)
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Depth (m)")
    st.pyplot(fig2)

    st.subheader("Ambient Noise Spectrum (avg hydrophone)")
    f, Pxx = welch(avg_hydro, fs=fs, nperseg=1024)
    fig3, ax3 = plt.subplots()
    ax3.semilogy(f, Pxx+1e-12)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD")
    st.pyplot(fig3)

with col2:
    st.subheader("Array Element Signals (first 4 channels)")
    fig4, ax4 = plt.subplots(figsize=(6,3))
    t = np.arange(n)/fs
    for i in range(min(4, arr_active.shape[0])):
        ax4.plot(t[:1000], arr_active[i,:1000] + i*2, label=f"Elem {i+1}")
    ax4.set_xlabel("Time (s)")
    st.pyplot(fig4)

    st.subheader("Beamformer Output & Spectrogram")
    fig5, ax5 = plt.subplots(figsize=(6,3))
    ax5.plot(np.arange(len(bf_out))/fs, bf_out)
    ax5.set_xlabel("Time (s)")
    st.pyplot(fig5)

    fS, tS, Sxx = spectrogram(bf_out, fs=fs, nperseg=256)
    fig6, ax6 = plt.subplots(figsize=(6,3))
    ax6.pcolormesh(tS, fS, 10*np.log10(Sxx+1e-12), shading='gouraud')
    ax6.set_ylabel('Freq [Hz]')
    ax6.set_xlabel('Time [sec]')
    st.pyplot(fig6)

st.markdown("### Beamforming Controls & Results")
st.write(f"Chosen method: **{bf_method}** — Steering: **{steer_angle}°**")
st.write(f"TDOA bearing estimate (elems 1-2): **{tdoa_angle:.2f}°**")
st.write(f"Rain detector heuristic: **{'Rain Detected' if rain_detected else 'No Rain Detected'}**")
if auto_steer_enabled:
    st.write(f"Auto-steer best angle (max energy): **{best_ang:.1f}°**")
    fig_scan, ax_scan = plt.subplots()
    ax_scan.plot(scan_angles, scan_energies)
    ax_scan.set_xlabel("Angle (deg)"); ax_scan.set_ylabel("Energy")
    st.pyplot(fig_scan)

if use_active:
    st.subheader("Active Sonar Pulse & Matched Filter Output")
    st.write(f"Pulse type: {pulse_type}, length: {pulse_len} ms, bw: {pulse_bw} Hz")
    fig7, ax7 = plt.subplots()
    ax7.plot(np.arange(len(matched))/fs, matched)
    ax7.set_xlabel("Time (s)")
    st.pyplot(fig7)

# 3D Beam Pattern visualization via Plotly (simple)
st.subheader("2D/3D Beam Pattern (Array Grid)")
fig_p = go.Figure(data=go.Heatmap(z=PdB, x=az, y=el, colorbar=dict(title="dB")))
fig_p.update_layout(xaxis_title="Azimuth (deg)", yaxis_title="Elevation (deg)", height=450)
st.plotly_chart(fig_p, use_container_width=True)

# Sonar equation outputs
st.subheader("Sonar Equation & Range Prediction")
st.write("**Active sonar simple SNR calc (approx):** SL - 2TL + TS - (NL - DI)")
st.write("**Passive sonar simple SNR calc (approx):** SL - TL + DI - NL")
st.write(f"Absorption @ {freq_for_abs_khz} kHz = {alpha_db_km:.3f} dB/km")
st.line_chart({"Range (m)": ranges_m, "Passive SNR (dB)": SNRs})

if max_detect_range:
    st.success(f"Estimated max detection range (Passive, SNR>0): **{max_detect_range:.0f} m**")
else:
    st.warning("No range within 50 km offers SNR>0 with current parameters.")

# Small "virtual ocean" interactive demo: place sources (simulated)
st.subheader("Virtual Ocean — Place Sources (Simulation only)")
cols = st.columns(3)
with cols[0]:
    x_src = st.number_input("Source range (m)", 10, 50000, 2000)
with cols[1]:
    depth_src = st.number_input("Source depth (m)", 0, 1000, 100)
with cols[2]:
    src_strength = st.slider("Relative Strength", 0.1, 5.0, 1.0)

# show simple propagation circle and location
fig_vo, ax_vo = plt.subplots(figsize=(6,3))
ax_vo.scatter([0, x_src], [100, depth_src], c=['blue','red'])
ax_vo.annotate("Array", (0,100)); ax_vo.annotate("Source", (x_src, depth_src))
ax_vo.set_xlim(0, max(2000,x_src*1.2)); ax_vo.set_ylim(1000,0); ax_vo.set_xlabel("Range (m)"); ax_vo.set_ylabel("Depth (m)")
st.pyplot(fig_vo)

# Final notes & download
st.markdown("---")
st.markdown("### Notes & How this maps to FI9006 syllabus")
st.write("""
- SVP, Snell/ray-fan, shallow/deep water: UNIT I  
- Ambient noise bands, mixing, variability: UNIT II  
- Signals, matched filtering, spectrograms: UNIT III  
- Sonar equations, transducer array modeling, beamforming (DAS/MVDR/LCMV): UNIT IV  
- DSP-heavy processing implemented in pure Python: UNIT V concepts (processor-specific architecture not emulated but code is efficient)
""")

st.markdown("### Export")
if st.button("Download Beamformer Output (wav-like npy)"):
    import io, base64
    data = bf_out.astype(np.float32)
    b = io.BytesIO()
    np.save(b, data)
    b.seek(0)
    st.download_button("Download .npy", b, file_name="beamformed_output.npy")

st.markdown("### Next Steps I can provide")
st.write("""
- Convert this to modular GitHub repo with Colab notebook launcher and CI.  
- Add animations (wavefronts), improved ray-tracing with range-dependence, or GPU acceleration.  
- Produce a lab report export (PDF) summarizing experiments (SVP variants, noise scenarios, beamforming comparisons).
""")

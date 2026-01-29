import numpy as np
import matplotlib.pyplot as plt


def generate_multipath_channel(bw_hz, subcarrier_space, n_subcarriers, coherence_bw_subcarriers, n_realizations=1):
    """
    Generates a time-invariant frequency-selective multipath fading channel.

    Args:
        bw_hz (float): Total Bandwidth in Hz (e.g., 20e6).
        subcarrier_space (float): Sub-carrier spacing in Hz (e.g., 0.48e6).
        n_subcarriers (int): Number of sub-carriers (e.g., 40).
        coherence_bw_subcarriers (float): Coherence bandwidth in number of sub-carriers (e.g., 10).
        n_realizations (int): Number of independent channels to generate (for statistical average).

    Returns:
        H_freq (numpy array): Frequency domain channel gains (complex). Shape: (n_realizations, n_subcarriers)
        h_time (numpy array): Time domain impulse response (complex).
    """

    # 1. Calculate Physical Parameters
    # subcarrier_spacing = bw_hz / n_subcarriers
    scs = subcarrier_space
    ts = 1 / (subcarrier_space * 64)  # Sampling time duration

    # Calculate required RMS Delay Spread based on Coherence Bandwidth
    # Rule of thumb: Bc ≈ 1 / (5 * tau_rms)
    # Therefore: tau_rms ≈ 1 / (5 * Bc)
    coher_bw = coherence_bw_subcarriers * scs
    tau_rms = 1 / (5 * coher_bw)

    print(f"--- Simulation Parameters ---")
    print(f"Bandwidth: {bw_hz / 1e6} MHz")
    print(f"Sub-carriers: {n_subcarriers}")
    print(f"Sub-carrier Spacing: {scs / 1e3} kHz")
    print(f"Target Coherence BW: {coherence_bw_subcarriers} sub-carriers ({coher_bw / 1e3} kHz)")
    print(f"Calculated RMS Delay Spread: {tau_rms * 1e9:.2f} ns")

    # 2. Define Power Delay Profile (PDP)
    # We use an Exponential Decay profile common for indoor/in-vehicle multipath
    # We need enough taps to capture the decay. 10 * tau_rms is usually sufficient.
    max_delay = 10 * tau_rms
    n_taps = int(np.ceil(max_delay / ts))

    # Ensure we don't have more taps than subcarriers (simulation constraint)
    # Ideally, n_subcarriers should be >= n_taps to avoid aliasing,
    n_taps = min(n_taps, n_subcarriers)

    tau = np.arange(n_taps) * ts
    pdp = np.exp(-tau / tau_rms)

    # Normalize PDP energy to 1
    pdp = pdp / np.sum(pdp)

    # 3. Generate Channel Coefficients (Rayleigh Fading)
    # h_time = (Real + j*Imag) * sqrt(Power)
    # We generate Circular Symmetric Complex Gaussian noise
    h_time = np.zeros((n_realizations, n_taps), dtype=complex)

    for i in range(n_realizations):
        real_part = np.random.randn(n_taps)
        imag_part = np.random.randn(n_taps)
        # Scale by PDP power
        # We multiply by sqrt(pdp/2) because variance is split between real and imag
        h_time[i, :] = (real_part + 1j * imag_part) * np.sqrt(pdp / 2)

    # 4. Convert to Frequency Domain (FFT)
    # We perform an N-point FFT. If taps < N, it zero-pads automatically.
    H_freq = np.fft.fft(h_time, n=n_subcarriers, axis=1)

    return H_freq, h_time, tau

def generate_flat_channel(n_subcarriers, n_realizations=1):
    """
    Generates a flat fading channel (frequency non-selective).

    Args:
        n_subcarriers (int): Number of sub-carriers.
        n_realizations (int): Number of independent channels to generate.

    Returns:
        H_freq (numpy array): Frequency domain channel gains (complex). Shape: (n_realizations, n_subcarriers)
        h_time (numpy array): Time domain impulse response (complex).
    """

    h_real = np.random.randn(n_realizations, 1)
    h_imag = np.random.randn(n_realizations, 1)

    # h ~ CN(0,1) -> Real and Imag parts each ~ N(0, 0.5)
    h_time = (h_real + 1j * h_imag) * np.sqrt(0.5)

    # FFT: impulse response [h, 0, 0, ..., 0] in time domain -> [H, H, H, ...] in frequency domain
    H_freq = np.repeat(h_time, n_subcarriers)

    return H_freq, h_time


if __name__ == "__main__":
    seed_value = 1
    np.random.seed(seed_value)

    # --- Configuration based on user request ---
    BW = 20e6  # 20 MHz
    SCS = 0.48e6  # 480 kHz
    N_SC = 8  # 40 Sub-carriers
    COHERENCE_SC = 16  # 10 Sub-carriers per coherence bandwidth

    # Generate the channel
    H_est, h_impulse, delays = generate_multipath_channel(BW, SCS, N_SC, COHERENCE_SC)

    # --- Visualization ---
    plt.figure(figsize=(12, 8))

    # 1. Power Delay Profile (Time Domain)
    plt.subplot(2, 1, 1)
    plt.stem(delays * 1e9, np.abs(h_impulse[0]), basefmt=" ")
    plt.title('Channel Impulse Response (Time Domain)')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Magnitude |h(t)|')
    plt.grid(True, alpha=0.3)

    # 2. Channel Frequency Response (Frequency Domain)
    plt.subplot(2, 1, 2)
    freqs = np.linspace(0, BW, N_SC) / 1e6  # MHz scale for plotting
    # FFT shift for plotting 0Hz in center
    H_shifted = np.fft.fftshift(H_est[0])
    H_mag_db = 20 * np.log10(np.abs(H_shifted) + 1e-12)  # Avoid log(0)

    plt.plot(freqs, H_mag_db, 'b-', marker='o', linewidth=2, label='Channel Gain')

    # Add visual markers for Coherence Bandwidth
    center_idx = N_SC // 2
    plt.axvspan(freqs[max(center_idx - int(COHERENCE_SC/2) + 1, 0)], freqs[min(center_idx + int(COHERENCE_SC/2), N_SC-1)],
                color='red', alpha=0.1, label='Approx Coherence BW Span')

    plt.title(f'Channel Frequency Response (Frequency Domain)')
    plt.xlabel('Carrier frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
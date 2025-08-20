import numpy as np

def fft1D(u_ic_values, x_coords, n_freq_to_keep):
    """
    Performs a 1D Fast Fourier Transform on initial condition data.
    
    Returns the top `n_freq_to_keep` frequencies and their corresponding
    cosine (a_coeffs) and sine (b_coeffs) coefficients.
    """
    n_points = len(u_ic_values)
    if n_points == 0:
        return np.array([]), np.array([]), np.array([])
        
    # Perform FFT
    fft_coeffs = np.fft.rfft(u_ic_values)
    # Calculate corresponding frequencies (not angular frequencies yet)
    spacing = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    frequencies = np.fft.rfftfreq(n_points, d=spacing)

    # Get the indices of the top N frequencies by magnitude (excluding DC component)
    magnitudes = np.abs(fft_coeffs)
    # We ignore the 0-th index (DC component) for mode selection
    top_indices = np.argsort(magnitudes[1:])[-n_freq_to_keep:] + 1
    top_indices = np.sort(top_indices)

    # Select the top frequencies and corresponding coefficients
    selected_freqs = frequencies[top_indices]
    selected_coeffs = fft_coeffs[top_indices]

    # Convert to cosine and sine coefficients
    # a_n = 2/N * Re(Y_n), b_n = -2/N * Im(Y_n)
    a_coeffs = 2 * np.real(selected_coeffs) / n_points
    b_coeffs = -2 * np.imag(selected_coeffs) / n_points
    
    # Convert frequencies to angular frequencies (k = 2 * pi * f)
    angular_frequencies = 2 * np.pi * selected_freqs

    return angular_frequencies, a_coeffs, b_coeffs
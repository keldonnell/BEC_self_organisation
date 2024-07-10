
import numpy as np
from scipy.signal import find_peaks
from scipy import integrate, stats, constants


def find_first_harmonic(fft_data, fft_freq_data, minimum_height = 0.00000003):
    
    # Find all peaks
    peaks, _ = find_peaks(fft_data, height = minimum_height)
    
    # Get peak amplitudes
    peak_amplitudes = fft_data[peaks]
    
    # Sort peak indices based on their amplitudes in descending order
    sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]
    
    if len(sorted_peak_indices) == 0:
        first_harmonic_amplitude = 0
        first_harmonic_frequency = 0
    else:
        # The highest peak should be the 1st harmonic
        first_harmonic_index = peaks[sorted_peak_indices[0]]
        first_harmonic_amplitude = fft_data[first_harmonic_index]
        first_harmonic_frequency = fft_freq_data[first_harmonic_index]

   
    return first_harmonic_frequency, first_harmonic_amplitude


def integrate_higher_modes(ft_data, freq_vals, first_harmonic_freq, harmonic_fraction=1):

    # Find the index of the first harmonic
    first_harmonic_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    
    # Find the second harmonic (assuming it's approximately double the frequency of the first)
    second_harmonic_freq = 2 * first_harmonic_freq
    second_harmonic_idx = np.argmin(np.abs(freq_vals - second_harmonic_freq))
    
    # Calculate the frequency difference between harmonics
    freq_diff = freq_vals[second_harmonic_idx] - freq_vals[first_harmonic_idx]
    
    # Set the integration width to a fraction of this difference
    integration_width = freq_diff * harmonic_fraction
    
    # Define the upper limit of the first harmonic integration window
    first_harmonic_upper_limit = first_harmonic_freq + integration_width / 2
    
    # Find the index corresponding to this upper limit
    upper_idx = np.searchsorted(freq_vals, first_harmonic_upper_limit)

    
    # Perform the integration for all higher modes
    area_higher_modes = integrate.simps(ft_data[upper_idx:], freq_vals[upper_idx:]) * 2
    
    return area_higher_modes


def integrate_first_harmonic_fwhm(ft_data, freq_vals, first_harmonic_freq, width_factor=2):
    # Find the index of the first harmonic peak
    peak_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    peak_value = ft_data[peak_idx]

    # Find half maximum value
    half_max = peak_value / 2

    # Find indices where the data crosses half maximum
    left_idx = peak_idx
    while left_idx > 0 and ft_data[left_idx] > half_max:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(ft_data) - 1 and ft_data[right_idx] > half_max:
        right_idx += 1

    # Calculate FWHM
    fwhm = freq_vals[right_idx] - freq_vals[left_idx]

    # Set integration limits
    integration_width = fwhm * width_factor
    lower_limit = first_harmonic_freq - integration_width / 2
    upper_limit = first_harmonic_freq + integration_width / 2

    # Find indices for integration
    lower_idx = np.searchsorted(freq_vals, lower_limit)
    upper_idx = np.searchsorted(freq_vals, upper_limit)

    # Perform the integration
    area = integrate.simps(ft_data[lower_idx:upper_idx], freq_vals[lower_idx:upper_idx])

    return area, fwhm


def integrate_first_harmonic(ft_data, freq_vals, first_harmonic_freq, harmonic_fraction=0.75):
    # Find the index of the first harmonic
    first_harmonic_idx = np.argmin(np.abs(freq_vals - first_harmonic_freq))
    
    # Find the second harmonic (assuming it's approximately double the frequency of the first)
    second_harmonic_freq = 2 * first_harmonic_freq
    second_harmonic_idx = np.argmin(np.abs(freq_vals - second_harmonic_freq))
    
    # Calculate the frequency difference between harmonics
    freq_diff = freq_vals[second_harmonic_idx] - freq_vals[first_harmonic_idx]
    
    # Set the integration width to a fraction of this difference
    integration_width = freq_diff * harmonic_fraction
    
    # Define integration limits
    lower_limit = first_harmonic_freq - integration_width / 2
    upper_limit = first_harmonic_freq + integration_width / 2
    
    # Find indices corresponding to these limits
    lower_idx = np.searchsorted(freq_vals, lower_limit)
    upper_idx = np.searchsorted(freq_vals, upper_limit)
    
    # Perform the integration
    area = integrate.simps(ft_data[lower_idx:upper_idx], freq_vals[lower_idx:upper_idx]) * 2
    
    return area


def analyse_fourier_data(sorted_files, freq_vals, norm_factor, is_temporal_ft, cut_index = None):
    """
    Analyze Fourier data from a list of files.

    :param sorted_files: List of file paths to analyze
    :param x_index: Index of x values in the data files
    :param num_crit: Critical number for normalization
    :return: Dictionary containing analysis results
    """

    if is_temporal_ft and cut_index == None:
        raise Exception("You must specify a cut index for temporal fourier transform data")

    first_mode_ft_peaks_amp = []
    first_mode_ft_peaks_freq = []
    first_mode_ft_peak_area = []
    higher_modes_ft_peak_area = []

    freq_vals = np.abs(freq_vals[:len(freq_vals)//2])
    
    for file in sorted_files:
        # Load the file
        data = np.loadtxt(file)

        if is_temporal_ft:
            psi_cut_vals = data[:, cut_index]
        else:
            max_index = np.argmax(data[:, (len(data[0,:]) // 2)])
            psi_cut_vals = data[max_index, 1:]


        fft_psi_vals = np.fft.fft(psi_cut_vals, norm="forward")
        fft_psi_vals = np.abs(fft_psi_vals[:len(fft_psi_vals)//2]) / (norm_factor)



        first_harmonic_frequency, first_harmonic_amplitude = find_first_harmonic(fft_psi_vals, freq_vals)
        first_mode_ft_peaks_amp.append(first_harmonic_amplitude)
        first_mode_ft_peaks_freq.append(first_harmonic_frequency)

        #first_mode_area = integrate_first_harmonic_fwhm(fft_psi_vals, freq_vals, first_harmonic_frequency, width_factor=4)[0]
        first_mode_area = integrate_first_harmonic(fft_psi_vals, freq_vals, first_harmonic_frequency)
        first_mode_ft_peak_area.append(first_mode_area)

        higher_mode_area = integrate_higher_modes(fft_psi_vals, freq_vals, first_harmonic_frequency)
        higher_modes_ft_peak_area.append(higher_mode_area)

    
    return {
        'first_mode_ft_peaks_amp': first_mode_ft_peaks_amp,
        'first_mode_ft_peaks_freq': first_mode_ft_peaks_freq,
        'first_mode_ft_peak_area': first_mode_ft_peak_area,
        'higher_modes_ft_peak_area': higher_modes_ft_peak_area
    }

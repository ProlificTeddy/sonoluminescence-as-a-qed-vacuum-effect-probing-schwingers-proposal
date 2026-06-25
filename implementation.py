import numpy as np
import torch

def calculate_bogolubov_coefficients(initial_radius, final_radius, frequency_range, speed_of_collapse):
    """
    Calculate the Bogolubov coefficients for photon production due to the dynamic Casimir effect.
    
    Parameters:
        initial_radius (float): Initial radius of the bubble.
        final_radius (float): Final radius of the bubble.
        frequency_range (torch.Tensor): Range of frequencies to calculate coefficients for.
        speed_of_collapse (float): Speed of bubble collapse.
    
    Returns:
        alpha (torch.Tensor): Bogolubov alpha coefficients.
        beta (torch.Tensor): Bogolubov beta coefficients.
    """
    # Calculate the change in the dielectric boundary
    delta_radius = initial_radius - final_radius
    
    # Calculate the Bogolubov coefficients
    alpha = torch.exp(-1j * frequency_range * delta_radius / speed_of_collapse)
    beta = torch.sqrt(1 - torch.abs(alpha)**2)
    
    return alpha, beta

def calculate_photon_spectrum(beta, frequency_range):
    """
    Calculate the photon spectrum from the Bogolubov beta coefficients.
    
    Parameters:
        beta (torch.Tensor): Bogolubov beta coefficients.
        frequency_range (torch.Tensor): Range of frequencies.
    
    Returns:
        spectrum (torch.Tensor): Photon spectrum.
    """
    # Photon spectrum is proportional to |beta|^2
    spectrum = torch.abs(beta)**2 * frequency_range
    return spectrum

def calculate_total_energy(spectrum, frequency_range):
    """
    Calculate the total energy emitted based on the photon spectrum.
    
    Parameters:
        spectrum (torch.Tensor): Photon spectrum.
        frequency_range (torch.Tensor): Range of frequencies.
    
    Returns:
        total_energy (float): Total energy emitted.
    """
    # Integrate the spectrum over the frequency range
    total_energy = torch.trapz(spectrum, frequency_range)
    return total_energy.item()

if __name__ == '__main__':
    # Define parameters
    initial_radius = 5.0  # Initial radius of the bubble (arbitrary units)
    final_radius = 1.0    # Final radius of the bubble (arbitrary units)
    speed_of_collapse = 0.1  # Speed of collapse (arbitrary units)
    frequency_range = torch.linspace(0.1, 10.0, 1000)  # Frequency range (arbitrary units)
    
    # Calculate Bogolubov coefficients
    alpha, beta = calculate_bogolubov_coefficients(initial_radius, final_radius, frequency_range, speed_of_collapse)
    
    # Calculate photon spectrum
    spectrum = calculate_photon_spectrum(beta, frequency_range)
    
    # Calculate total energy emitted
    total_energy = calculate_total_energy(spectrum, frequency_range)
    
    # Print results
    print("Photon Spectrum (first 10 values):", spectrum[:10].numpy())
    print("Total Energy Emitted:", total_energy)
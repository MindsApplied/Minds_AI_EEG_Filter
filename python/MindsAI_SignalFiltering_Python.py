import os
import ctypes
import numpy as np
import time
import msvcrt  # Windows-only for detecting key press
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ********************************************
# * Variable Initialization *
# Requirements: MindsAI DLL & License Key, Brainflow SDK and Board DLL
board_id = BoardIds.SYNTHETIC_BOARD.value  # Brainflow Board of Choice
mindsai_license = ""  # License Key provided by MindsApplied
seconds_window = 10  # Number of seconds per window
segments = 1  # Number of windows being input (real-time uses 1 window per update)
lambda_value = 1e-8  # Lambda to be updated based on data size
# Sampling Rate and Channel Number inherited from BrainFlow
# ********************************************

# Path to MindsAI DLL
dll_path = os.path.join(os.path.dirname(__file__), "MindsAI.dll")
MindsAI_dll = ctypes.CDLL(dll_path)

# Declare argument and return types
MindsAI_dll.InitializeMindsAILicense.argtypes = [ctypes.c_char_p]
MindsAI_dll.InitializeMindsAILicense.restype = ctypes.c_bool

MindsAI_dll.ApplyMindsAIFilter.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
]

MindsAI_dll.GetMindsAILicenseMessage.restype = ctypes.c_char_p

# Initialize the license (must happen once before filtering)
def initialize_mindsai_license(license_key: str):
    return MindsAI_dll.InitializeMindsAILicense(license_key.encode("utf-8"))


def get_license_message():
    return MindsAI_dll.GetMindsAILicenseMessage().decode("utf-8")


def apply_mindsai_filter(eeg_data_2d, segments, channels, timepoints, lambda_val):
    reshaped_input = eeg_data_2d.T.flatten().astype(np.float64)
    output_flat = np.zeros((channels * timepoints), dtype=np.float64)
    MindsAI_dll.ApplyMindsAIFilter(reshaped_input, segments, timepoints, channels, lambda_val, output_flat)
    return output_flat.reshape((channels, timepoints))


# Initialize BrainFlow
params = BrainFlowInputParams()
board = BoardShim(board_id, params)

BoardShim.enable_dev_board_logger()
board.prepare_session()
board.start_stream()

sampling_rate = BoardShim.get_sampling_rate(board_id)
data_points = sampling_rate * seconds_window

eeg_channels = BoardShim.get_eeg_channels(board_id)
channels = len(eeg_channels)

license_ok = initialize_mindsai_license(mindsai_license)
print("License initialized:", "Success" if license_ok else "Failure")
print("License status:", get_license_message())
print(f"Streaming EEG data every {seconds_window} seconds. Press any key to stop...")

try:
    while not msvcrt.kbhit():
        time.sleep(seconds_window)
        data = board.get_current_board_data(data_points)
        eeg_data = np.array([data[ch] for ch in eeg_channels])

        if license_ok:
            filtered = apply_mindsai_filter(eeg_data, segments, channels, eeg_data.shape[1], lambda_value)
            for ch in range(filtered.shape[0]):
                print(f"[Filtered] Channel {ch + 1} Last Value: {filtered[ch, -1]:.4f}")
        else:
            for ch in range(eeg_data.shape[0]):
                print(f"[Raw] Channel {ch + 1} Last Value: {eeg_data[ch, -1]:.4f}")
finally:
    board.stop_stream()
    board.release_session()

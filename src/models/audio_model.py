from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class AudioModel:
    """
    Model layer responsible for interfacing with Windows system audio hardware via Pycaw.
    """
    def __init__(self):
        # Initialize audio speakers device
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume_control = interface.QueryInterface(IAudioEndpointVolume)

        # Retrieve system volume range (typically [-65.25, 0.0] dB)
        self.vol_range = self.volume_control.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]

    def set_volume_db(self, vol_db: float) -> None:
        """
        Set master volume using decibel value within hardware bounds.
        """
        clamped_vol = max(self.min_vol, min(self.max_vol, vol_db))
        self.volume_control.SetMasterVolumeLevel(clamped_vol, None)

    def get_min_vol(self) -> float:
        """Get minimum decibel volume limit."""
        return self.min_vol

    def get_max_vol(self) -> float:
        """Get maximum decibel volume limit."""
        return self.max_vol

import pandas as pd
import numpy as np
import pvlib


class SolarModel:
    """
    Computes hourly DC power output for a rooftop solar installation.

    The model chain:
      1. Compute sun position for each hour (altitude, azimuth)
      2. Transpose GHI to plane-of-array (POA) irradiance using the
         isotropic sky model
      3. Estimate cell temperature from ambient temperature + POA
      4. Apply the photovoltaic efficiency equation with temperature correction

    Parameters
    ----------
    panel_area_m2 : float
        Total panel surface area in square meters
    efficiency : float
        Panel efficiency as a fraction (e.g. 0.20 for 20%)
    temp_coefficient : float
        Power loss per degree C above reference (e.g. 0.004)
    tilt_deg : float
        Panel tilt angle in degrees from horizontal
    azimuth_deg : float
        Panel azimuth in degrees (180 = south-facing)
    latitude : float
        Site latitude in decimal degrees
    longitude : float
        Site longitude in decimal degrees
    altitude_m : float
        Site elevation in meters above sea level
    """

    def __init__(
        self,
        panel_area_m2: float,
        efficiency: float,
        temp_coefficient: float,
        tilt_deg: float,
        azimuth_deg: float,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ):
        self.panel_area_m2 = panel_area_m2
        self.efficiency = efficiency
        self.temp_coefficient = temp_coefficient
        self.tilt_deg = tilt_deg
        self.azimuth_deg = azimuth_deg

        # pvlib location object — bundles site coordinates for sun position calcs
        self.location = pvlib.location.Location(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude_m,
            tz="America/Los_Angeles",
        )
    
    def generate(self, climate_df: pd.DataFrame) -> pd.Series:
        """
        Compute hourly DC power output from climate data.

        Parameters
        ----------
        climate_df : pd.DataFrame
            DataFrame with a DatetimeIndex (hourly, timezone-aware) and columns:
            - GHI : Global Horizontal Irradiance (W/m²)
            - DNI : Direct Normal Irradiance (W/m²)
            - DHI : Diffuse Horizontal Irradiance (W/m²)
            - temp_air : Ambient temperature (°C)

        Returns
        -------
        pd.Series
            Hourly DC power output in kilowatts (kW)
        """

        # Step 1: Compute sun position for every hour in the dataset
        # Returns solar zenith, azimuth, altitude for each timestamp
        solar_position = self.location.get_solarposition(climate_df.index)

        # Step 2: Transpose GHI to plane-of-array (POA) irradiance
        # POA = how much irradiance actually hits our tilted, oriented panel
        # We use the isotropic sky model — diffuse light arrives equally from all sky directions
        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt_deg,
            surface_azimuth=self.azimuth_deg,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=climate_df["DNI"],
            ghi=climate_df["GHI"],
            dhi=climate_df["DHI"],
            model="isotropic",
        )

        # poa_global is the total irradiance on the panel surface in W/m²
        poa = poa_irradiance["poa_global"].fillna(0)

        # Step 3: Estimate cell temperature
        # Panels heat up beyond ambient temperature when irradiance is high
        # pvlib's SAPM model estimates this from POA + ambient temp + wind
        # We use a simplified open-rack glass/glass configuration
        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=climate_df["temp_air"],
            wind_speed=0,  # conservative assumption — no wind cooling
            a=-3.56,       # module-level empirical constants for open-rack glass
            b=-0.075,
            deltaT=3,
        )

        # Step 4: Apply the power equation with temperature correction
        # P_dc = Area × efficiency × POA × [1 - γ(T_cell - T_ref)]
        # This is exactly the equation we derived in the math foundation
        t_ref = 25.0  # standard test condition reference temperature in °C
        temp_correction = 1 - self.temp_coefficient * (cell_temp - t_ref)

        # Power in watts
        p_dc_watts = self.panel_area_m2 * self.efficiency * poa * temp_correction

        # Convert to kilowatts and clip negatives
        # (temperature correction can go slightly negative at extreme cold — physically meaningless)
        p_dc_kw = (p_dc_watts / 1000).clip(lower=0)

        p_dc_kw.name = "solar_kw"
        return p_dc_kw



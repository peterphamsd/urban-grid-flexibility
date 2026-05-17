import numpy as np
import pandas as pd


class LoadModel:
    """
    Generates hourly residential electricity load profiles for San Diego.

    Uses a baseline hourly profile shaped around real consumption patterns,
    applies seasonal scaling, and adds Gaussian perturbation to simulate
    day-to-day variation.

    Parameters
    ----------
    annual_kwh : float
        Target annual consumption in kWh (typical SD home: ~6,000 kWh)
    perturbation_std : float
        Standard deviation of hourly random variation (default 0.10 = 10%)
    random_seed : int or None
        Seed for reproducibility across runs
    """
    # -------------------------------------------------------------------------
    # NOTE: Synthetic baseline
    # The hourly shape and seasonal multipliers below are synthetic — derived
    # from the known duck curve pattern for San Diego residential consumption
    # but not sourced from measured data. A future version will replace these
    # with real NREL ResStock hourly profiles for San Diego climate zone 6.
    # -------------------------------------------------------------------------


    # Normalized 24-hour baseline shape — sums to 24
    # Represents the typical hourly consumption pattern for a San Diego home
    # Values are relative weights, scaled to match annual_kwh target
    HOURLY_SHAPE = np.array([
        0.60, 0.55, 0.52, 0.50, 0.52, 0.58,  # 12am - 5am (overnight low)
        0.75, 0.90, 0.85, 0.80, 0.82, 0.85,  # 6am - 11am (morning ramp)
        0.88, 0.85, 0.83, 0.85, 0.90, 1.10,  # 12pm - 5pm (midday, rising)
        1.40, 1.50, 1.45, 1.30, 1.10, 0.80,  # 6pm - 11pm (evening peak)
    ])

    # Multiplier for seasons
    SEASONAL_MULTIPLIERS = {
        1:  0.85,   # January
        2:  0.83,   # February
        3:  0.87,   # March
        4:  0.88,   # April
        5:  0.95,   # May
        6:  1.05,   # June
        7:  1.18,   # July
        8:  1.25,   # August  ← peak AC month
        9:  1.10,   # September
        10: 0.95,   # October
        11: 0.88,   # November
        12: 0.87,   # December
    }

    def __init__(
        self,
        annual_kwh: float = 6000.0,
        perturbation_std: float = 0.10,
        random_seed: int = None,
    ):
        self.annual_kwh = annual_kwh
        self.perturbation_std = perturbation_std
        self.rng = np.random.default_rng(random_seed)

        # Scale hourly shape to match annual target
        # HOURLY_SHAPE sums to around 24, multiply by 365 days = 8760 hours per year
        raw_annual = self.HOURLY_SHAPE.mean() * 8760
        self.scale_factor = annual_kwh / raw_annual

    def generate(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Generate an hourly load profile for the given time index.

        Parameters-
        index : pd.DatetimeIndex
            Hourly timestamps to generate load for (timezone-aware)

        Returns

        pd.Series
            Hourly load in kilowatts (kW)
        """

        loads = []

        for timestamp in index:
            # Base load for this hour of day
            hour = timestamp.hour
            base = self.HOURLY_SHAPE[hour] * self.scale_factor

            # Apply seasonal multiplier
            month = timestamp.month
            seasonal = self.SEASONAL_MULTIPLIERS[month]

            # Apply Gaussian perturbation
            # L(h) = L_base(h) * seasonal * (1 + ε)
            # ε ~ Normal(0, perturbation_std)
            epsilon = self.rng.normal(0, self.perturbation_std)
            load = base * seasonal * (1 + epsilon)

            # Load can't be negative
            loads.append(max(load, 0.0))

        return pd.Series(loads, index=index, name="load_kw")
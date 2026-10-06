"""Physical constants (SI units unless stated)."""

MU_EARTH = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
R_EARTH_EQ = 6378137.0             # WGS84 equatorial radius [m]
WGS84_F = 1.0 / 298.257223563      # WGS84 flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
J2 = 1.08262668e-3                 # Earth J2 zonal harmonic
OMEGA_EARTH = 7.2921150e-5         # Earth rotation rate [rad/s]
C_LIGHT = 299792458.0              # speed of light [m/s]
K_BOLTZMANN_DB = -228.6            # Boltzmann constant [dBW/K/Hz]

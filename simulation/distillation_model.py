import numpy as np
from simulation.thermodynamics import Thermodynamics, SYSTEMS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DistillationColumn:
    def __init__(self, n_stages=20, feed_stage=10, pressure=101.325, system='benzene-toluene', model=None, industrial_data=None):
        self.n_stages = n_stages
        self.feed_stage = feed_stage
        self.pressure = pressure
        self.system = system
        self.industrial_data = industrial_data
        self.use_empirical = (system == 'industrial-custom' and industrial_data is not None)

        if self.use_empirical:
            self._build_empirical_model()
            self.components = [
                {'name': 'Light Key', 'boiling_point': 80, 'cp_liquid': 100, 'cp_vapor': 80, 'hvap': 30000, 'mw': 50},
                {'name': 'Heavy Key', 'boiling_point': 110, 'cp_liquid': 120, 'cp_vapor': 90, 'hvap': 35000, 'mw': 80}
            ]
            self.thermo = None
        else:
            system_data = SYSTEMS.get(system)
            if system_data is None:
                print("Available systems:")
                for name in SYSTEMS.keys():
                    print(f"  - {name}")
                raise ValueError(f"System '{system}' not found!")

            self.components = system_data['components']()
            self.system = system
            self.model = model if model else system_data['model']
            self.thermo = Thermodynamics(self.components, model=self.model)

    def _build_empirical_model(self):
        """Build KNN empirical model from industrial data"""
        df = self.industrial_data.copy()

        # Required columns check
        required = ['reflux_ratio', 'feed_rate', 'feed_composition', 
                   'top_purity', 'bottom_purity', 'reboiler_duty', 'condenser_duty']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Industrial data missing columns: {missing}")

        # Features
        X = df[['reflux_ratio', 'feed_rate', 'feed_composition']].values
        y = df[['top_purity', 'bottom_purity', 'reboiler_duty', 'condenser_duty']].values

        # Scale features for better KNN performance
        self.empirical_scaler = StandardScaler()
        X_scaled = self.empirical_scaler.fit_transform(X)

        # Use KNN with k=3 (or min of data points)
        k = min(5, len(df))
        self.empirical_model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        self.empirical_model.fit(X_scaled, y)

        # Store data ranges for validation
        self.data_ranges = {
            'reflux_ratio': (df['reflux_ratio'].min(), df['reflux_ratio'].max()),
            'feed_rate': (df['feed_rate'].min(), df['feed_rate'].max()),
            'feed_composition': (df['feed_composition'].min(), df['feed_composition'].max())
        }

        # Calculate average distillate/bottoms split from data
        self.avg_split = 0.5  # Default
        if 'distillate_rate' in df.columns and 'feed_rate' in df.columns:
            self.avg_split = (df['distillate_rate'] / df['feed_rate']).mean()

        # Store average temperatures if available
        self.avg_temps = {
            'top': df['top_temperature'].mean() if 'top_temperature' in df.columns else 80,
            'bottom': df['bottom_temperature'].mean() if 'bottom_temperature' in df.columns else 110,
            'feed': df['feed_temperature'].mean() if 'feed_temperature' in df.columns else 95
        }

        # Calculate effective relative volatility from purity data
        self.effective_alpha = self._estimate_alpha_from_data(df)

    def _estimate_alpha_from_data(self, df):
        """Estimate effective relative volatility from operating data"""
        # Using Fenske equation approximation: xD/(1-xD) = alpha^N * xB/(1-xB)
        # For typical operation, alpha ≈ (xD/(1-xD) / (xB/(1-xB)))^(1/N)
        alphas = []
        for _, row in df.iterrows():
            xd = row['top_purity']
            xb = row['bottom_purity']
            if xb > 0.001 and xd < 0.999 and xb < xd:
                try:
                    alpha = (xd/(1-xd) / (xb/(1-xb))) ** (1/self.n_stages)
                    if 1 < alpha < 50:
                        alphas.append(alpha)
                except:
                    pass
        return np.median(alphas) if alphas else 2.5

    def _validate_empirical_input(self, reflux_ratio, feed_rate, feed_composition):
        """Warn if inputs are outside training data range"""
        warnings = []
        for param, (min_val, max_val) in self.data_ranges.items():
            val = locals()[param]
            if val < min_val or val > max_val:
                warnings.append(f"{param}={val:.3f} outside training range [{min_val:.3f}, {max_val:.3f}]")
        return warnings

    def calculate_stage(self, x_liquid, T_guess=None):
        if self.use_empirical:
            return None, None, None
        T, y_vapor = self.thermo.bubble_point(x_liquid, self.pressure)
        props = self._get_properties(T, x_liquid, y_vapor)
        return T, y_vapor, props

    def _get_properties(self, T, x, y):
        if self.use_empirical or self.thermo is None:
            return {}
        return {
            'temperature': T,
            'liquid_enthalpy': self.thermo.liquid_enthalpy(T, x),
            'vapor_enthalpy': self.thermo.vapor_enthalpy(T, y),
            'heat_of_vaporization': self.thermo.vapor_enthalpy(T, y) - self.thermo.liquid_enthalpy(T, x),
            'vapor_pressure': [self.thermo.antoine(T, i) for i in range(self.thermo.n_comp)]
        }

    def simulate(self, reflux_ratio, feed_rate, feed_composition):
        if self.use_empirical:
            return self._simulate_empirical(reflux_ratio, feed_rate, feed_composition)
        else:
            return self._simulate_physical(reflux_ratio, feed_rate, feed_composition)

    def _simulate_empirical(self, reflux_ratio, feed_rate, feed_composition):
        """Empirical simulation using trained KNN model"""
        warnings = self._validate_empirical_input(reflux_ratio, feed_rate, feed_composition)

        X = np.array([[reflux_ratio, feed_rate, feed_composition]])
        X_scaled = self.empirical_scaler.transform(X)
        y_pred = self.empirical_model.predict(X_scaled)[0]

        top_purity, bottom_purity, reboiler_duty, condenser_duty = y_pred

        # Estimate flow rates
        D = feed_rate * self.avg_split
        B = feed_rate - D
        L = reflux_ratio * D
        V = L + D

        # Estimate temperatures (from data averages or simple correlation)
        # Higher reflux = lower top temperature (more reflux = more cooling)
        top_temp = self.avg_temps['top'] - 2 * (reflux_ratio - 3)
        bottom_temp = self.avg_temps['bottom'] + 1 * (reflux_ratio - 3)
        feed_temp = self.avg_temps['feed']

        return {
            'top_purity': top_purity,
            'bottom_purity': bottom_purity,
            'reboiler_duty': reboiler_duty,
            'condenser_duty': condenser_duty,
            'distillate_rate': D,
            'bottoms_rate': B,
            'top_temperature': top_temp,
            'bottom_temperature': bottom_temp,
            'feed_temperature': feed_temp,
            'relative_volatility': self.effective_alpha,
            'empirical_mode': True,
            'empirical_warnings': warnings,
            'n_stages': self.n_stages,
            'feed_stage': self.feed_stage
        }

    def _simulate_physical(self, reflux_ratio, feed_rate, feed_composition):
        """Original physical simulation with improved purity calculation"""
        z_feed = np.array([feed_composition, 1 - feed_composition])
        T_feed, y_feed = self.thermo.bubble_point(z_feed, self.pressure)
        alpha = self._calculate_alpha(T_feed)

        # ==================== IMPROVED PURITY CALCULATION ====================
        # Use Gilliland correlation to estimate actual N vs N_min at given R
        # Then solve for achievable purity

        # First estimate: use Fenske for minimum stages at infinite reflux
        # Then apply Gilliland to find actual separation at finite R

        # For simplicity, use exponential approach to equilibrium:
        # xD approaches equilibrium value as N increases and R increases

        # Equilibrium purity at total reflux (Fenske limit)
        x_d_eq = min(feed_composition * alpha / (1 + feed_composition * (alpha - 1)), 0.9999)
        x_b_eq = max(feed_composition / (alpha * (1 - feed_composition) + feed_composition), 0.0001)

        # Effect of finite reflux: separation efficiency factor
        # Higher R = closer to total reflux = higher purity
        # More stages = closer to equilibrium = higher purity
        R_min = (x_d_eq - y_feed[0]) / (y_feed[0] - x_d_eq) if y_feed[0] > x_d_eq else 0.5
        R_min = max(R_min, 0.1)

        # Gilliland correlation: Y = 1 - exp((1 + 54.4X)/(11 + 117.2X) * (X-1)/X)
        # where X = (R - R_min)/(R + 1), Y = (N - N_min)/(N + 1)
        X_gill = (reflux_ratio - R_min) / (reflux_ratio + 1)
        X_gill = max(X_gill, 0.001)

        if X_gill < 0.01:
            # Very close to minimum reflux, poor separation
            efficiency = 0.1
        else:
            Y_gill = 1 - np.exp((1 + 54.4 * X_gill) / (11 + 117.2 * X_gill) * (X_gill - 1) / X_gill)
            efficiency = max(0.1, min(0.99, Y_gill))

        # Apply stage efficiency (Murphree ~0.6-0.8 for real trays)
        stage_efficiency = 0.7
        actual_stages = self.n_stages * stage_efficiency

        # Combined effect: more stages and higher R improve purity
        stage_factor = 1 - np.exp(-actual_stages / 10)  # Approaches 1 with many stages
        reflux_factor = efficiency  # From Gilliland

        # Interpolate between feed composition and equilibrium purity
        x_d = feed_composition + (x_d_eq - feed_composition) * stage_factor * reflux_factor
        x_b = feed_composition - (feed_composition - x_b_eq) * stage_factor * reflux_factor

        # Clamp to physical limits
        x_d = min(max(x_d, feed_composition), 0.9999)
        x_b = max(min(x_b, feed_composition), 0.0001)

        # Ensure x_d > x_b
        if x_d <= x_b + 0.01:
            x_d = min(feed_composition + 0.1, 0.95)
            x_b = max(feed_composition - 0.1, 0.05)
        # ==================== END IMPROVED PURITY ====================

        D = feed_rate * (feed_composition - x_b) / (x_d - x_b)
        B = feed_rate - D
        L = reflux_ratio * D
        V = L + D

        T_top, _ = self.thermo.bubble_point(np.array([x_d, 1-x_d]), self.pressure)
        T_bottom, _ = self.thermo.bubble_point(np.array([x_b, 1-x_b]), self.pressure)

        Hvap_top = self.thermo.vapor_enthalpy(T_top, np.array([x_d, 1-x_d])) - self.thermo.liquid_enthalpy(T_top, np.array([x_d, 1-x_d]))
        Hvap_bottom = self.thermo.vapor_enthalpy(T_bottom, np.array([x_b, 1-x_b])) - self.thermo.liquid_enthalpy(T_bottom, np.array([x_b, 1-x_b]))

        reboiler_duty = V * abs(Hvap_bottom) / 3600
        condenser_duty = V * abs(Hvap_top) / 3600

        return {
            'top_purity': x_d,
            'bottom_purity': x_b,
            'reboiler_duty': reboiler_duty,
            'condenser_duty': condenser_duty,
            'distillate_rate': D,
            'bottoms_rate': B,
            'top_temperature': T_top,
            'bottom_temperature': T_bottom,
            'feed_temperature': T_feed,
            'relative_volatility': alpha,
            'empirical_mode': False,
            'empirical_warnings': [],
            'n_stages': self.n_stages,
            'feed_stage': self.feed_stage,
            'gilliland_efficiency': efficiency,
            'stage_efficiency': stage_efficiency
        }

    def _calculate_alpha(self, T):
        if self.thermo is None:
            return self.effective_alpha if hasattr(self, 'effective_alpha') else 2.5
        P_sat = [self.thermo.antoine(T, i) for i in range(self.thermo.n_comp)]
        return P_sat[0] / P_sat[1]

    def get_thermo_properties(self, T, x):
        if self.use_empirical or self.thermo is None:
            return {}
        _, y = self.thermo.bubble_point(x, self.pressure)
        return self._get_properties(T, x, y)

    def get_system_info(self):
        if self.use_empirical:
            return {
                'name': 'industrial-custom',
                'components': ['Light Key', 'Heavy Key'],
                'model': 'empirical (KNN)',
                'description': f'Empirical model from {len(self.industrial_data)} plant data points',
                'data_points': len(self.industrial_data),
                'effective_alpha': self.effective_alpha
            }
        return {
            'name': self.system,
            'components': [c['name'] for c in self.components],
            'model': self.model,
            'description': SYSTEMS[self.system]['description']
        }

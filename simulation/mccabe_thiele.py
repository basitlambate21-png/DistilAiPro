import numpy as np
from simulation.thermodynamics import Thermodynamics

class McCabeThiele:
    def __init__(self, thermo, pressure=101.325, empirical_data=None):
        self.thermo = thermo
        self.pressure = pressure
        self.empirical_data = empirical_data
        self.use_empirical = empirical_data is not None

        if self.use_empirical:
            self._build_empirical_vle()

    def _build_empirical_vle(self):
        """Build empirical equilibrium curve from plant data"""
        df = self.empirical_data

        # If we have top/bottom purity data, we can estimate equilibrium curve
        # Using the relationship: y = alpha*x / (1 + (alpha-1)*x)
        # We estimate alpha from operating data

        alphas = []
        for _, row in df.iterrows():
            xd = row.get('top_purity', 0.5)
            xb = row.get('bottom_purity', 0.5)
            if xd > xb and xb > 0.01 and xd < 0.99:
                # Approximate alpha from separation
                alpha_est = (xd/(1-xd)) / (xb/(1-xb))
                if 1 < alpha_est < 50:
                    alphas.append(alpha_est)

        self.empirical_alpha = np.median(alphas) if alphas else 2.5

        # Build empirical equilibrium curve
        self.empirical_x = np.linspace(0.001, 0.999, 100)
        alpha = self.empirical_alpha
        self.empirical_y = alpha * self.empirical_x / (1 + (alpha - 1) * self.empirical_x)

    def equilibrium_curve(self, n_points=100):
        if self.use_empirical:
            return self.empirical_x, self.empirical_y, np.zeros_like(self.empirical_x)

        x_values = np.linspace(0.001, 0.999, n_points)
        y_values = []
        T_values = []

        for x in x_values:
            x_arr = np.array([x, 1-x])
            T, y = self.thermo.bubble_point(x_arr, self.pressure)
            y_values.append(y[0])
            T_values.append(T)

        return np.array(x_values), np.array(y_values), np.array(T_values)

    def operating_lines(self, R, x_d, x_b, z_f, q):
        slope_rect = R / (R + 1)
        intercept_rect = x_d / (R + 1)

        if abs(q - 1.0) < 0.001:
            x_int = z_f
            y_int = slope_rect * x_int + intercept_rect
            slope_q = float('inf')
            intercept_q = z_f
        else:
            m_q = q / (q - 1)
            b_q = -z_f / (q - 1)
            x_int = (intercept_rect - b_q) / (m_q - slope_rect)
            y_int = slope_rect * x_int + intercept_rect
            slope_q = m_q
            intercept_q = b_q

        if abs(x_int - x_b) < 0.0001:
            slope_strip = 1.0
            intercept_strip = 0
        else:
            slope_strip = (y_int - x_b) / (x_int - x_b)
            intercept_strip = x_b - slope_strip * x_b

        return {
            'rectifying': {'slope': slope_rect, 'intercept': intercept_rect},
            'stripping': {'slope': slope_strip, 'intercept': intercept_strip},
            'q_line': {'slope': slope_q, 'intercept': intercept_q},
            'intersection': (x_int, y_int)
        }

    def step_off_stages(self, R, x_d, x_b, z_f, q, max_stages=100):
        x_eq, y_eq, _ = self.equilibrium_curve()
        op = self.operating_lines(R, x_d, x_b, z_f, q)

        steps_x = []
        steps_y = []

        # Start at (x_d, x_d) on 45-degree line
        y_current = x_d

        feed_stage = 1
        n_stages = 0

        for stage in range(max_stages):
            # Step 1: Find x on equilibrium curve where y = y_current (horizontal line)
            x_current = np.interp(y_current, y_eq, x_eq)

            # Clamp
            x_current = max(min(x_current, x_d), x_b)

            steps_x.append(x_current)
            steps_y.append(y_current)
            n_stages += 1

            # Check if reached bottoms
            if x_current <= x_b + 0.001:
                break

            # Step 2: Find y on operating line at x_current (vertical line)
            if x_current >= op['intersection'][0]:
                y_next = op['rectifying']['slope'] * x_current + op['rectifying']['intercept']
            else:
                y_next = op['stripping']['slope'] * x_current + op['stripping']['intercept']

            # Clamp
            y_next = max(y_next, x_b)
            y_next = min(y_next, x_d)

            steps_x.append(x_current)
            steps_y.append(y_next)

            # Detect feed stage
            if x_current >= op['intersection'][0] and y_next < op['intersection'][1]:
                feed_stage = n_stages

            # Check if reached bottoms
            if y_next <= x_b + 0.001:
                break

            # Check for stagnation
            if abs(y_next - y_current) < 0.0001 and stage > 5:
                break

            y_current = y_next

        return {
            'n_stages': n_stages,
            'stages_x': np.array(steps_x),
            'stages_y': np.array(steps_y),
            'feed_stage': feed_stage,
            'empirical_mode': self.use_empirical
        }

    def minimum_reflux(self, z_f, x_d, x_b, q):
        x_eq, y_eq, _ = self.equilibrium_curve()

        if abs(q - 1.0) < 0.001:
            x_p = z_f
            y_p = np.interp(x_p, x_eq, y_eq)
        else:
            m_q = q / (q - 1)
            b_q = -z_f / (q - 1)
            y_diff = y_eq - (m_q * x_eq + b_q)
            sign_changes = np.where(np.diff(np.sign(y_diff)))[0]

            if len(sign_changes) > 0:
                idx = sign_changes[0]
                x_p = x_eq[idx]
                y_p = y_eq[idx]
            else:
                x_p = z_f
                y_p = np.interp(z_f, x_eq, y_eq)

        if abs(x_d - x_p) < 0.0001:
            R_min = 10.0
        else:
            slope = (x_d - y_p) / (x_d - x_p)
            if slope >= 1.0:
                R_min = 10.0
            else:
                R_min = slope / (1.0 - slope)

        return max(R_min, 0.5)

    def minimum_stages(self, x_d, x_b, alpha_avg):
        if x_d <= x_b or x_d >= 1 or x_b <= 0:
            return 100
        return np.log((x_d / (1 - x_d)) * ((1 - x_b) / x_b)) / np.log(alpha_avg)

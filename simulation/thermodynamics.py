import numpy as np
from scipy.optimize import brentq

class Thermodynamics:
    def __init__(self, components, model='ideal'):
        self.components = components
        self.n_comp = len(components)
        self.model = model

    def antoine(self, T, comp_idx):
        A, B, C = self.components[comp_idx]['antoine']
        P_sat = 10 ** (A - B / (T + C))
        return P_sat * 133.322 / 1000

    def bubble_point(self, x, P_total):
        def objective(T):
            P_sat = np.array([self.antoine(T, i) for i in range(self.n_comp)])
            if self.model == 'ideal':
                gamma = np.ones(self.n_comp)
            else:
                gamma = self.activity_coefficient(x, T)
            K = gamma * P_sat / P_total
            return np.sum(K * x) - 1.0

        # ==================== DYNAMIC TEMPERATURE BOUNDS ====================
        T_guesses = []
        for i, comp in enumerate(self.components):
            if 'boiling_point' in comp:
                T_guesses.append(comp['boiling_point'])
            else:
                # Estimate from Antoine equation
                for T_try in [-50, -20, 0, 20, 50, 100, 150, 200, 250, 300]:
                    try:
                        if self.antoine(T_try, i) >= P_total:
                            T_guesses.append(T_try)
                            break
                    except:
                        continue
                else:
                    T_guesses.append(100)

        T_low = min(T_guesses) - 100
        T_high = max(T_guesses) + 100
        T_low = max(T_low, -150)
        T_high = min(T_high, 500)

        # Ensure different signs for brentq
        f_low = objective(T_low)
        f_high = objective(T_high)

        attempts = 0
        while f_low * f_high > 0 and attempts < 20:
            T_low -= 30
            f_low = objective(T_low)
            if f_low * f_high > 0:
                T_high += 30
                f_high = objective(T_high)
            attempts += 1

        if f_low * f_high > 0:
            raise ValueError(f"Cannot find bubble point: f({T_low})={f_low:.4f}, f({T_high})={f_high:.4f}")

        T_bubble = brentq(objective, T_low, T_high)
        # ==================== END DYNAMIC BOUNDS ====================

        P_sat = np.array([self.antoine(T_bubble, i) for i in range(self.n_comp)])
        gamma = np.ones(self.n_comp) if self.model == 'ideal' else self.activity_coefficient(x, T_bubble)
        K = gamma * P_sat / P_total
        y = K * x
        return T_bubble, y / np.sum(y)

    def dew_point(self, y, P_total):
        def objective(T):
            P_sat = np.array([self.antoine(T, i) for i in range(self.n_comp)])
            K = P_sat / P_total
            return np.sum(y / K) - 1.0

        T_guesses = []
        for i, comp in enumerate(self.components):
            if 'boiling_point' in comp:
                T_guesses.append(comp['boiling_point'])
            else:
                for T_try in [-50, -20, 0, 20, 50, 100, 150, 200, 250, 300]:
                    try:
                        if self.antoine(T_try, i) >= P_total:
                            T_guesses.append(T_try)
                            break
                    except:
                        continue
                else:
                    T_guesses.append(100)

        T_low = max(min(T_guesses) - 100, -150)
        T_high = min(max(T_guesses) + 100, 500)

        f_low = objective(T_low)
        f_high = objective(T_high)

        attempts = 0
        while f_low * f_high > 0 and attempts < 20:
            T_low -= 30
            f_low = objective(T_low)
            if f_low * f_high > 0:
                T_high += 30
                f_high = objective(T_high)
            attempts += 1

        if f_low * f_high > 0:
            raise ValueError(f"Cannot find dew point: f({T_low})={f_low:.4f}, f({T_high})={f_high:.4f}")

        T_dew = brentq(objective, T_low, T_high)

        P_sat = np.array([self.antoine(T_dew, i) for i in range(self.n_comp)])
        x = y * P_total / P_sat
        return T_dew, x / np.sum(x)

    def activity_coefficient(self, x, T):
        if self.model == 'wilson':
            return self.wilson_model(x, T)
        elif self.model == 'nrtl':
            return self.nrtl_model(x, T)
        return np.ones(self.n_comp)

    def wilson_model(self, x, T):
        if 'wilson_params' in self.components[0]:
            Lambda = self.components[0]['wilson_params']
        else:
            return np.ones(self.n_comp)

        gamma = np.ones(self.n_comp)
        for i in range(self.n_comp):
            sum1 = np.sum(x * Lambda[i, :])
            sum2 = 0
            for j in range(self.n_comp):
                sum2 += x[j] * Lambda[j, i] / np.sum(x * Lambda[j, :])
            gamma[i] = np.exp(-np.log(sum1) + 1 - sum1 + sum2)
        return gamma

    def nrtl_model(self, x, T):
        if 'nrtl_params' in self.components[0]:
            tau = self.components[0]['nrtl_params']
        else:
            return np.ones(self.n_comp)

        G = np.exp(-0.3 * tau)
        gamma = np.ones(self.n_comp)
        for i in range(self.n_comp):
            sum1 = np.sum(x * G[:, i])
            sum2 = np.sum(x * tau[:, i] * G[:, i])
            sum3 = 0
            for j in range(self.n_comp):
                sum3 += x[j] * G[i, j] / np.sum(x * G[j, :]) * (tau[i, j] - np.sum(x * tau[:, j] * G[:, j]) / np.sum(x * G[j, :]))
            gamma[i] = np.exp(sum2 / sum1 + sum3)
        return gamma

    def liquid_enthalpy(self, T, x):
        H = 0
        for i, comp in enumerate(self.components):
            H += x[i] * comp['cp_liquid'] * (T - 25)
        return H

    def vapor_enthalpy(self, T, y):
        H = 0
        for i, comp in enumerate(self.components):
            H += y[i] * (comp['cp_liquid'] * (T - 25) + comp['hvap'] + comp['cp_vapor'] * (T - 25))
        return H


# ==================== SYSTEM DATABASE ====================

def get_benzene_toluene():
    return [
        {'name': 'Benzene', 'antoine': [6.90565, 1211.033, 220.79], 'cp_liquid': 137, 'cp_vapor': 82, 'hvap': 30770, 'mw': 78.11, 'boiling_point': 80.1},
        {'name': 'Toluene', 'antoine': [6.95464, 1344.8, 219.48], 'cp_liquid': 157, 'cp_vapor': 106, 'hvap': 33180, 'mw': 92.14, 'boiling_point': 110.6}
    ]

def get_ethanol_water():
    return [
        {'name': 'Ethanol', 'antoine': [8.20417, 1642.89, 230.300], 'cp_liquid': 112, 'cp_vapor': 65, 'hvap': 38560, 'mw': 46.07, 'boiling_point': 78.4,
         'wilson_params': np.array([[1.0, 0.25], [0.35, 1.0]]),
         'nrtl_params': np.array([[0.0, 1.5], [0.8, 0.0]])},
        {'name': 'Water', 'antoine': [8.07131, 1730.63, 233.426], 'cp_liquid': 75, 'cp_vapor': 34, 'hvap': 40650, 'mw': 18.015, 'boiling_point': 100.0,
         'wilson_params': np.array([[1.0, 0.35], [0.25, 1.0]]),
         'nrtl_params': np.array([[0.0, 0.8], [1.5, 0.0]])}
    ]

def get_methanol_water():
    return [
        {'name': 'Methanol', 'antoine': [8.08097, 1582.271, 239.726], 'cp_liquid': 81, 'cp_vapor': 44, 'hvap': 35270, 'mw': 32.04, 'boiling_point': 64.7,
         'wilson_params': np.array([[1.0, 0.3], [0.4, 1.0]])},
        {'name': 'Water', 'antoine': [8.07131, 1730.63, 233.426], 'cp_liquid': 75, 'cp_vapor': 34, 'hvap': 40650, 'mw': 18.015, 'boiling_point': 100.0,
         'wilson_params': np.array([[1.0, 0.4], [0.3, 1.0]])}
    ]

def get_hexane_heptane():
    return [
        {'name': 'Hexane', 'antoine': [6.87787, 1171.53, 224.366], 'cp_liquid': 195, 'cp_vapor': 137, 'hvap': 28850, 'mw': 86.18, 'boiling_point': 68.7},
        {'name': 'Heptane', 'antoine': [6.90240, 1268.115, 216.900], 'cp_liquid': 224, 'cp_vapor': 165, 'hvap': 31770, 'mw': 100.21, 'boiling_point': 98.4}
    ]

def get_acetone_water():
    return [
        {'name': 'Acetone', 'antoine': [7.11714, 1210.595, 229.664], 'cp_liquid': 125, 'cp_vapor': 75, 'hvap': 30200, 'mw': 58.08, 'boiling_point': 56.0,
         'wilson_params': np.array([[1.0, 0.15], [0.5, 1.0]])},
        {'name': 'Water', 'antoine': [8.07131, 1730.63, 233.426], 'cp_liquid': 75, 'cp_vapor': 34, 'hvap': 40650, 'mw': 18.015, 'boiling_point': 100.0,
         'wilson_params': np.array([[1.0, 0.5], [0.15, 1.0]])}
    ]

def get_propane_propylene():
    return [
        {'name': 'Propane', 'antoine': [6.80334, 804.07, 247.04], 'cp_liquid': 98, 'cp_vapor': 74, 'hvap': 18470, 'mw': 44.10, 'boiling_point': -42.1},
        {'name': 'Propylene', 'antoine': [6.80168, 785.63, 247.24], 'cp_liquid': 92, 'cp_vapor': 68, 'hvap': 18420, 'mw': 42.08, 'boiling_point': -47.6}
    ]

def get_isopropanol_water():
    return [
        {'name': 'Isopropanol', 'antoine': [8.11778, 1580.92, 219.61], 'cp_liquid': 163, 'cp_vapor': 90, 'hvap': 39800, 'mw': 60.10, 'boiling_point': 82.6,
         'wilson_params': np.array([[1.0, 0.2], [0.45, 1.0]])},
        {'name': 'Water', 'antoine': [8.07131, 1730.63, 233.426], 'cp_liquid': 75, 'cp_vapor': 34, 'hvap': 40650, 'mw': 18.015, 'boiling_point': 100.0,
         'wilson_params': np.array([[1.0, 0.45], [0.2, 1.0]])}
    ]

def get_chloroform_acetone():
    return [
        {'name': 'Chloroform', 'antoine': [6.95465, 1170.966, 226.232], 'cp_liquid': 114, 'cp_vapor': 72, 'hvap': 29300, 'mw': 119.38, 'boiling_point': 61.2,
         'wilson_params': np.array([[1.0, 0.6], [0.4, 1.0]])},
        {'name': 'Acetone', 'antoine': [7.11714, 1210.595, 229.664], 'cp_liquid': 125, 'cp_vapor': 75, 'hvap': 30200, 'mw': 58.08, 'boiling_point': 56.0,
         'wilson_params': np.array([[1.0, 0.4], [0.6, 1.0]])}
    ]

# Dictionary of all systems
SYSTEMS = {
    'benzene-toluene': {'components': get_benzene_toluene, 'model': 'ideal', 'description': 'Ideal system, close boiling'},
    'ethanol-water': {'components': get_ethanol_water, 'model': 'wilson', 'description': 'Azeotrope at 89.4% ethanol'},
    'methanol-water': {'components': get_methanol_water, 'model': 'wilson', 'description': 'Azeotrope at ~98% methanol'},
    'hexane-heptane': {'components': get_hexane_heptane, 'model': 'ideal', 'description': 'Ideal, petroleum refining'},
    'acetone-water': {'components': get_acetone_water, 'model': 'wilson', 'description': 'Partial miscibility'},
    'propane-propylene': {'components': get_propane_propylene, 'model': 'ideal', 'description': 'Close boiling, LPG separation'},
    'isopropanol-water': {'components': get_isopropanol_water, 'model': 'wilson', 'description': 'Azeotrope at 87.7%'},
    'chloroform-acetone': {'components': get_chloroform_acetone, 'model': 'wilson', 'description': 'Maximum boiling azeotrope'}
}

def get_system_list():
    print("Available Systems:")
    print("-" * 50)
    for name, info in SYSTEMS.items():
        print(f"  {name:20s} - {info['description']}")
    print("-" * 50)

def get_system(name):
    if name not in SYSTEMS:
        print(f"System '{name}' not found!")
        get_system_list()
        return None
    return SYSTEMS[name]['components'](), SYSTEMS[name]['model']

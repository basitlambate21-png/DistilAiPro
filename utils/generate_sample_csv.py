import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.distillation_model import DistillationColumn
import pandas as pd
import numpy as np

def generate_sample_csv(filename='sample_distillation_data.csv', n_samples=1000):
    column = DistillationColumn(n_stages=20, feed_stage=10)
    
    data = []
    for _ in range(n_samples):
        rr = np.random.uniform(1.5, 8.0)
        fr = np.random.uniform(30, 180)
        fc = np.random.uniform(0.2, 0.8)
        
        result = column.simulate(rr, fr, fc)
        
        data.append({
            'reflux_ratio': round(rr, 2),
            'feed_rate': round(fr, 1),
            'feed_composition': round(fc, 3),
            'top_purity': round(result['top_purity'], 4),
            'bottom_purity': round(result['bottom_purity'], 4),
            'reboiler_duty': round(result['reboiler_duty'], 2),
            'condenser_duty': round(result['condenser_duty'], 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {n_samples} rows")
    return df

if __name__ == "__main__":
    generate_sample_csv()
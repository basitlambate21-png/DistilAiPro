import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.distillation_model import DistillationColumn

def generate_dataset(n_samples=5000, system='benzene-toluene', model='ideal'):
    print(f"Generating {n_samples} samples for {system} ({model})...")
    
    np.random.seed(42)
    reflux_ratios = np.random.uniform(1.0, 8.0, n_samples)
    feed_rates = np.random.uniform(20, 200, n_samples)
    feed_compositions = np.random.uniform(0.1, 0.9, n_samples)
    
    column = DistillationColumn(n_stages=20, feed_stage=10, system=system, model=model)
    results = []
    
    for i, (rr, fr, fc) in enumerate(zip(reflux_ratios, feed_rates, feed_compositions)):
        output = column.simulate(rr, fr, fc)
        results.append({
            'reflux_ratio': rr,
            'feed_rate': fr,
            'feed_composition': fc,
            'top_purity': output['top_purity'],
            'bottom_purity': output['bottom_purity'],
            'reboiler_duty': output['reboiler_duty'],
            'condenser_duty': output['condenser_duty']
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_samples}")
    
    df = pd.DataFrame(results)
    filename = f'data/training_data_{system}_{model}.csv'
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}!")
    return df

if __name__ == "__main__":
    generate_dataset()
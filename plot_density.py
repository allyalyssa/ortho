import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

df = pd.read_csv('data/stimuli_density.csv')

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'related': 'blue', 'unrelated': 'red'}

for condition in ['related', 'unrelated']:
    subset = df[df['condition'] == condition]
    ax.scatter(subset['density'], range(len(subset)), 
               c=colors[condition], alpha=0.6, label=condition, s=50)

ax.set_xlabel('Orthographic Neighborhood Density')
ax.set_ylabel('Word Index')
ax.set_title('Orthographic Density by Condition')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/density_scatter.png', dpi=300)
logger.info('Saved figures/density_scatter.png')

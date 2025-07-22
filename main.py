import pandas as pd
import matplotlib.pyplot as plt


cols = ['user','activity','timestamp','x','y','z']

df = (pd.read_csv(
        'WISDM_ar_v1.1_raw.txt',              # adjust path if needed
        header=None,
        names=cols,
        sep=r',|\s*;',                    # split on commas or the trailing semicolon
        engine='python'
     )
     .dropna(subset=cols)                 # drop incomplete rows
)

# Ensure numeric dtypes
df[['x','y','z']] = df[['x','y','z']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['x','y','z'])

# Convert timestamp (µs) to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')

print(df.head())

ax = (df['activity']
      .value_counts()
      .sort_values()
      .plot(kind='barh', title='Training examples by activity type'))

ax.set_xlabel('Count')
plt.tight_layout()
plt.show()
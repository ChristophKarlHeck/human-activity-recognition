import pandas as pd
import matplotlib.pyplot as plt



cols = ['user', 'activity', 'timestamp', 'x', 'y', 'z']

df = (pd.read_csv(
        'WISDM_cleaned.txt',
        header=None,
        names=cols,
        sep=',',
        engine='python'
     )
     .dropna(subset=cols)
)

df[['x','y','z']] = df[['x','y','z']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['x','y','z'])

# Convert timestamp (µs) to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')

df['activity'] = df['activity'].str.lower()  # Lowercase for safety

df['activity'] = df['activity'].replace({
    'upstairs': 'other',
    'downstairs': 'other',
    'sitting': 'other',
    'standing': 'other'
})

print(df.head())

print(df['activity'].value_counts())

min_count = df['activity'].value_counts().min()

df_balanced = (
    df.groupby('activity')
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

ax = (df_balanced['activity']
      .value_counts()
      .sort_values()
      .plot(kind='barh', title='Training examples by activity type'))

ax.set_xlabel('Count')
plt.tight_layout()
plt.show()
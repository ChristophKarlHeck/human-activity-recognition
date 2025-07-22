import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

COLS = ['user', 'activity', 'timestamp', 'x', 'y', 'z']


def plot_random_user_timeseries(df: pd.DataFrame, seed: int | None = None) -> int:
    """
    Pick a random user, plot x/y/z vs. relative time (s).
    Color changes with activity labels: walking / running / other.
    Returns the chosen user id.
    """
    rng = np.random.default_rng(seed)
    user_id = rng.choice(df['user'].unique())
    sub = df[df['user'] == 33].sort_values('timestamp').copy()
    sub['t_rel'] = (sub['timestamp'] - sub['timestamp'].iloc[0]).dt.total_seconds()

    # Assign a color to each activity
    color_map = {'walking': 'tab:blue', 'running': 'tab:orange', 'other': 'tab:green'}
    # Ensure every label has a color (fallback)
    sub['color'] = sub['activity'].map(color_map).fillna('tab:gray')

    # Find contiguous segments where activity stays the same
    seg_id = (sub['activity'] != sub['activity'].shift()).cumsum()
    sub['segment'] = seg_id

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for i, axis_name in enumerate(['x', 'y', 'z']):
        for _, seg in sub.groupby('segment'):
            axes[i].plot(seg['t_rel'], seg[axis_name], color=seg['color'].iloc[0])
        axes[i].set_ylabel(axis_name)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'User {user_id} – accelerometer over time (color = activity)')

    # Legend
    legend_handles = [Patch(color=c, label=lbl) for lbl, c in color_map.items()]
    fig.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.show()
    return user_id

def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load WISDM file, fix dtypes, timestamps, and merge minor classes into 'other'."""
    df = (pd.read_csv(path, header=None, names=COLS, sep=',', engine='python')
            .dropna(subset=COLS))

    # numeric conversion
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['x', 'y', 'z'])

    # timestamp to datetime (microseconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # normalize labels
    df['activity'] = df['activity'].str.lower()
    df['activity'] = df['activity'].replace({
        'upstairs': 'other',
        'downstairs': 'other',
        'sitting': 'other',
        'standing': 'other'
    })

    # sort by user then time
    df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

    _ = plot_random_user_timeseries(df, seed=42)

    return df



def balance_by_downsampling(df: pd.DataFrame, label_col: str = 'activity',
                            random_state: int = 42) -> pd.DataFrame:
    """Downsample each class to the minority class size."""
    min_count = df[label_col].value_counts().min()
    return (df.groupby(label_col, group_keys=False)
              .sample(n=min_count, random_state=random_state))


def plot_activity(activity: str, df: pd.DataFrame, n: int = 500) -> None:
    """Plot first n samples of x/y/z for a given activity."""
    data = df[df['activity'] == activity][['x', 'y', 'z']].head(n)
    axes = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axes:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()


def plot_class_counts(df: pd.DataFrame, label_col: str = 'activity', title: str = '') -> None:
    ax = (df[label_col]
          .value_counts()
          .sort_values()
          .plot(kind='barh', title=title or f'Examples per {label_col}'))
    ax.set_xlabel('Count')
    plt.tight_layout()
    plt.show()


def get_frequency(df: pd.DataFrame, max_gap_s: float = 1.0) -> float:
    """
    Estimate sampling frequency by taking the median of small inter-sample gaps.
    Filters out long pauses between recordings.
    """
    df = df.sort_values(['user', 'timestamp'])
    dt = df.groupby('user')['timestamp'].diff().dt.total_seconds()

    # keep only realistic sample-to-sample gaps
    dt = dt[(dt > 0) & (dt < max_gap_s)]

    if dt.empty:
        raise ValueError("No valid deltas left after filtering. Check timestamp parsing or max_gap_s.")

    period = dt.median()
    freq = 1.0 / period
    print(f"period: {period:.6f} s")
    print(f"freq:   {freq:.2f} Hz")
    return freq


def main(show_plots: bool = False):
    data_path = 'WISDM_cleaned.txt'   # adjust if needed
    df = load_and_clean(data_path)

    print(df.head())
    print(df['activity'].value_counts())

    #_ = get_frequency(df)

    df_bal = balance_by_downsampling(df)

    if show_plots:
        plot_class_counts(df_bal, title='Training examples by activity type')
        plot_activity('jogging', df)
        plot_activity('walking', df)
        plot_activity('other', df)


if __name__ == '__main__':
    main()

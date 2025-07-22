# human-activity-recognition

## Data from 

```
@article{kwapisz2011activity,
  title={Activity recognition using cell phone accelerometers},
  author={Kwapisz, Jennifer R and Weiss, Gary M and Moore, Samuel A},
  journal={ACM SigKDD Explorations Newsletter},
  volume={12},
  number={2},
  pages={74--82},
  year={2011},
  publisher={ACM New York, NY, USA}
}
```

## Clean original file:

```
with open('WISDM_ar_v1.1_raw.txt', 'r') as infile, open('WISDM_cleaned.txt', 'w') as outfile:
    for line in infile:
        fields = [f.strip() for f in line.strip().replace(';','').split(',')]
        if len(fields) == 6:
            outfile.write(','.join(fields) + '\n')
```
import os

src = 'artifacts/BiGRU_Modle.keras'
dst = 'artifacts/BiGRU_Model.keras'

if os.path.exists(src):
    os.rename(src, dst)
    print(f'Renamed {src} to {dst}')
else:
    print(f'Source file {src} not found')
    # List artifacts
    for f in os.listdir('artifacts'):
        print(' -', f)
import os


path = 'coordinate_files/coordinate_files/test'
samples = os.listdir(path)
total = 0
for sample in samples:
    sample_path = os.path.join(path, sample)
    if os.path.isdir(sample_path):
        total += len(os.listdir(sample_path))
print(total)
from pathlib import Path,PurePosixPath
import zipfile
root=Path(__file__).resolve().parents[1]/'results/augsburg-seed42'
with zipfile.ZipFile(root/'predictions.zip') as z:
 for n in z.namelist():
  if PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts:raise ValueError('Unsafe result path')
 z.extractall(root)
print('Restored 82 prediction CSV files:',root/'evaluation')

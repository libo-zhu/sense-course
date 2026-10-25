"""Download only missing public data and extract the selected raw scenes."""
from pathlib import Path,PurePosixPath
import argparse,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['augsburg','houston2013','all'],default='augsburg');a=p.parse_args()
 datasets=['augsburg','houston2013'] if a.dataset=='all' else [a.dataset]
 raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
 for ds in datasets:
  mods=['hsi','sar','dsm'] if ds=='augsburg' else ['hsi','ms','lidar']
  needed=[ROOT/f'data/processed/{ds}/{m}/{m}_{kind}_train.mat' for m in mods for kind in ['X','Y']]
  if not all(x.exists() for x in needed):
   subprocess.run([sys.executable,str(ROOT/'scripts/download_data.py'),'--dataset',ds,'--profile','train'],check=True)
  folder='HS-SAR-DSM Augsburg' if ds=='augsburg' else 'HS-MS Houston2013'
  required=['data_HS_LR.mat','data_SAR_HR.mat','data_DSM.mat','TrainImage.mat','TestImage.mat'] if ds=='augsburg' else ['data_HS_LR.mat','data_MS_HR.mat','2013_IEEE_GRSS_DF_Contest_Samples_TR.tif','2013_IEEE_GRSS_DF_Contest_Samples_VA.tif']
  if not all((raw/'Datasets'/folder/n).exists() for n in required):
   subprocess.run([sys.executable,str(ROOT/'scripts/download_archive.py')],check=True)
   tar=shutil.which('bsdtar') or ('/usr/bin/tar' if sys.platform=='darwin' else None)
   if not tar:raise RuntimeError('RAR extraction requires bsdtar (libarchive-tools on Linux).')
   archive=raw/'S2FL_benchmarks.rar'
   names=subprocess.check_output([tar,'-tf',str(archive)],text=True).splitlines()
   if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
    raise ValueError('Unsafe archive member path')
   subprocess.run([tar,'-xf',str(archive),'-C',str(raw),'Datasets/'+folder],check=True)
  # Augsburg test patches are cropped lazily from the raw scene; no 4.54 GB expanded MAT.
  if ds=='houston2013' and not (ROOT/'data/processed/houston2013/hsi/hsi_X_test.mat').exists():
   subprocess.run([sys.executable,str(ROOT/'scripts/download_data.py'),'--dataset',ds,'--profile','small-test'],check=True)
  if ds=='augsburg' and not all((ROOT/f'data/processed/augsburg/{m}/{m}_Y_test.mat').exists() for m in ['sar','dsm']):
   subprocess.run([sys.executable,str(ROOT/'scripts/download_data.py'),'--dataset',ds,'--profile','small-test'],check=True)
  print(f'{ds}: data ready')
if __name__=='__main__':main()

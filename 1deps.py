import os
import sys
if os.path.exists('/home/tito'):
    sys.path.append('/home/tito/kaggle/icecube-neutrinos-in-deep-ice/notebook_graphnet/software/graphnet/src')
    KAGGLE_ENV = False
else:
    KAGGLE_ENV = True
    sys.path.append('/kaggle/working/software/graphnet/src')

if KAGGLE_ENV:
    # Move software to working disk
    !rm  -r software
    !scp -r /kaggle/input/graphnet-and-dependencies/software .

    # Install dependencies
    !pip install /kaggle/working/software/dependencies/torch-1.11.0+cu115-cp37-cp37m-linux_x86_64.whl
    !pip install /kaggle/working/software/dependencies/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
    !pip install /kaggle/working/software/dependencies/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
    !pip install /kaggle/working/software/dependencies/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
    !pip install /kaggle/working/software/dependencies/torch_geometric-2.0.4.tar.gz

    # Install GraphNeT
    !cd software/graphnet;pip install --no-index --find-links="/kaggle/working/software/dependencies" -e .[torch]
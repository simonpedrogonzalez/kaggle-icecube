uv pip install kaggle/working/software/dependencies/torch-1.11.0+cu115-cp37-cp37m-linux_x86_64.whl
uv pip install kaggle/working/software/dependencies/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
uv pip install kaggle/working/software/dependencies/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
uv pip install kaggle/working/software/dependencies/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
uv pip install kaggle/working/software/dependencies/torch_geometric-2.0.4.tar.gz
# cd kaggle/working/software/graphnet;uv pip install --no-index --find-links="../dependencies" -e .[torch]
# uv add pip
# cd kaggle/working/software/graphnet;/home/simon/repos/kaggle-icecube/.venv/bin/python3 -m pip install --no-index --find-links="../dependencies" -e .[torch]

sudo ln -s /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcusparse.so.12 /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcusparse.so.11


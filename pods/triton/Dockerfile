FROM nvcr.io/nvidia/tritonserver:25.04-py3

# Install PyTorch + PyG + XGBoost + cuDF into the Triton conda environment.
# model.py lives on NFS (model-repo-pvc) and is loaded by Triton Python backend
# at runtime — no Python model code is baked into this image.
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torch_geometric==2.6.1 \
    xgboost==3.0.2 \
    numpy \
    pyarrow

COPY pods/triton/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]

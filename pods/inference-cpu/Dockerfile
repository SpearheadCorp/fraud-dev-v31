FROM nvcr.io/nvidia/tritonserver:24.02-py3

COPY pods/inference-cpu/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]

FROM nvcr.io/nvidia/pytorch:22.10-py3
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
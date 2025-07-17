FROM ultralytics/ultralytics:latest
RUN apt update
RUN apt install -y nano wget
RUN pip install grpcio-tools lapx

# Pre-download YOLO11s model
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

COPY ../.. /workspace
WORKDIR /workspace
RUN cd /workspace/proto && bash ./generate.sh

CMD ["python", "./serving/yolo/yolo_service.py"]
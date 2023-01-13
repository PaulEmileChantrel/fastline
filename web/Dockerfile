FROM python:3.10.0-slim

COPY . /app
WORKDIR /app

RUN python3 -m venv /otp/fastlinevenv3

RUN /otp/fastlinevenv3/bin/pip3 install pip --upgrade && \
    /otp/fastlinevenv3/bin/pip3 install -r requirements.txt && \
    chmod +x entrypoints.sh

CMD ["/app/entrypoints.sh"]

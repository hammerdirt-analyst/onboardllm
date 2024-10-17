# use official nvidia cuda 12.x runtime image hd
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
LABEL authors="roger"

# install python 3.10 and pip hd
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# set working directory hd
WORKDIR /app

# copy requirements.txt to leverage docker cache hd
COPY requirements.txt .

# install dependencies inside the container hd
RUN pip3 install --no-cache-dir -r requirements.txt

# copy the rest of the application code hd
COPY main.py .

# expose port 6000 for the flask app hd
EXPOSE 6000

# run the app with gunicorn hd
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:6000", "main:app"]




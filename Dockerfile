FROM ubuntu:latest
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev libffi-dev python3-dev apt-utils bash git vim nano python3 \
    python3-venv python3-pip
RUN groupadd -g 1000 nasser
RUN useradd -u 1000 -ms /bin/bash -d /home/nasser -m -g nasser nasser
WORKDIR /home/nasser/diagnosis_spinal_abnormalities/
COPY --chown=nasser:nasser user_home_dir/diagnosis_spinal_abnormalities diagnosis_spinal_abnormalities
USER nasser
COPY user_home_dir/diagnosis_spinal_abnormalities/requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]

















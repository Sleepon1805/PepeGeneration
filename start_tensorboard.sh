#!/bin/bash

python -m webbrowser -t "http://localhost:6006/"
tensorboard --logdir "./lightning_logs/" --samples_per_plugin images=100
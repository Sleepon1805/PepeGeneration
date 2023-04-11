#!/bin/bash

python -m webbrowser -t "http://localhost:6006/"
tensorboard --logdir "./lightning_logs/"
#!/bin/sh

kill -9 $(lsof -t -i :8000)

nohup python app.py > /dev/null 2>&1 &

# Introduction

In this project, I implemented a simple automated trading system. It can collect data automatically, analyze data automatically, and trade automatically. This system is not perfect, it is rough from a technical point of view, and it is difficult to manage from a software engineering point of view. But through this project, I understood the working process of an automated trading system, and I also learned many new technologies.

# Environment

On the whole, it is a lightweight pyTorch environment based on Docker, in which the official pyTorch image is used. If you don't have the image, it will be downloaded automatically.

# Structure and tools

In data collection module I use selenium to simulate browser and to collect data. The models used to predict the price or to make decisions are written using pyTorch.

# Usage

clone the project and run the 'run.py'
```
python run.py
```



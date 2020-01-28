# Introduction

The purpose of creating this project is to design an efficient, distributed, scalable, and cross-language data transmission system. The system is designed based on Kafka, and the entire system is built on the basis of a kube cluster. The system will implement a unified data read and write interface, whether on the external network or the internal network. The system will also achieve standardized implementation and configuration of common services. In addition, the system should automatically scale up or down, and automatically load balance.

# Environment

On the whole, it is a lightweight pyTorch environment based on Docker, in which the official pyTorch image is used. If you don't have the image, it will be downloaded automatically.

# Structure and tools

In data collection module I use selenium to simulate browser and to collect data. The models used to predict the price or to make decisions are written using pyTorch.

![sys-structure](structure.png)

# Usage

clone the project and run the 'run.py'
```
python run.py
```



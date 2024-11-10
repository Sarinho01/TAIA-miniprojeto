# TAIA-miniprojeto-HO-ShipNet

A simple adapted implementation of the "HO-ShipNet" article for the mini-project in the Advanced AI Topics 1 course.

## Installation

Follow these steps to set up the project environment:

```bash
# Clone this repository
git clone https://github.com/Sarinho01/TAIA-miniprojeto-HO-ShipNet

# Navigate to the project directory
cd TAIA-miniprojeto-HO-ShipNet
```
- Download the dataset from https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery?select=shipsnet.json.

- Place the images from the shipsnet/shipsnet folder of the downloaded site into the dataset/AllShips folder. If it doesn't exist, create it.

```bash
# Run Docker commands:
docker build -t ho_shipnet .
docker run ho_shipnet
```

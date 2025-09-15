# bumblebee-tracking
Tracking and monitoring behaviour of bumblebees with QR codes and YOLO deteciton algorithm


## Installation
**This project requires `Python 3.10` as a minimum requirement**

Once you have cloned down this repository using `git clone`, cd into the app directory:

```bash
git clone git@github.com:pratorum/Bumblebee_tracking.git
cd Bumblebee_tracking
```

Create a virtual environment and install the package:

```
python3 -m venv venv
source venv/bin/activate
pip install -e
```

If you want to install the development tools as well, use the following commands if you are are using **bash**:

```
pip install -e .[dev, ci]
```

If you are using **zsh**:

```
pip install -e ".[dev, ci]"
```

To exit the virtual environment use `deactivate`.
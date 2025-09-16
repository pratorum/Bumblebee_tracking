# bumblebee-tracking
Tracking and monitoring behaviour of bumblebees with QR codes and YOLO deteciton algorithm

## Installation
To install, use:

```
pip install git+https://github.com/pratorum/Bumblebee_tracking.git
```

**This project requires `Python 3.10` as minimum**


## Developer Installation
If you want to contribute to the repository, install as follows:
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

If you want to create a virtualenv with a specific python version use <path-to-python> -m venv venv

If you want to install the development tools as well, use the following commands if you are are using **bash**:

```
pip install -e .[dev]
```

If you are using **zsh**:

```
pip install -e ".[dev]"
```

To exit the virtual environment use `deactivate`.
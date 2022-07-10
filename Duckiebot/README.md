# Template: Duckiebot-template

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `packages/duckiebot` of your new repository. The package contains:
* Main file is placed in `src/duckiebot.py`
* Library for communicating with the robot is in `include/my_package/lib_duckiebot.py`
* Roslaunch file in `launch/duckiebot.launch` for launching the main file

### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

### 6. Docker commands
#### Build on host 
    dts devel build -f -u duckiebot
#### Run on host
    docker run -it --rm  -v ~/Downloads/data:/data --name duckiebot duckiebot/dt-duckiebot-interface:v2-amd64
#### Build on duckie
    docker -H patka1.local build -t duckiebot:v1 .
#### Run on duckie
    docker -H patka1.local run -it --rm  -v /data:/data --name duckiebot --privileged --network=host duckiebot:v1
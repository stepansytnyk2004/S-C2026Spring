# S-C2026Spring
Sensors and Control Spring 2026 Project Assignment -> **Chosen Track: Track B**
#

Hello!

We are Team 7 for the Sensors and Control Integrated Project. Our team members are:

- Stepan Sytnyk (24960991)
- Sebastian De Vries (25297375)

For this project we have implemented a UR3 within MuJoCO that picks up a Candy Jar and places it into a basket. Through a wrist-mounted RGB-D camera, it is able to detect and move to the Candy Jar using PBVS. To prove functionality, the Candy Jar's location is randomized throughout each trial.

## What it does
- **Camera calibration** 
- **Depth localisation** 
- **Target filter** 
- **Joint control** 
- **Visual servoing** 
- **Evaluation** 

For a full list of requirements see the **requirements.txt**

To run the demo, run the following command: `python main.py` (add `--camera` for the live wrist camera view).

## Video

You can watch our demo video here: https://www.youtube.com/watch?v=5WBcgFY3TSw

## Contributions

Stepan: For this project I completed the RGBD wrist camera and calibration, depth perception and the joint model, LQR and LQG.

Sebastian: For this project I completed the inital scene setup and robot movement functions, the Kalman filter and the PBVS approach/pick
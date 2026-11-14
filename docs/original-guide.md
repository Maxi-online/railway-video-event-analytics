<h1 align=center> RCD Industrial Video Analytics - AI</h1>

## Engineering review

Pipelline video analysts with stream reading, specification, tracking and subject-matter rules of events. Kafka and backend client link personnel processing to external events and hazard zone configuration.

### Code navigation

- [main.py](main.py)
- [src/services/inferencer/inferencer.py](src/services/inferencer/inferencer.py)
- [src/services/tracker/tracker.py](src/services/tracker/tracker.py)
- [src/services/danger_zones_reader.py](src/services/danger_zones_reader.py)
- [src/config/config.yaml](src/config/config.yaml)

### Reproduction and evaluation

Use authorized or synthetic video and pre-set trajectory to demonstrate. Compare times and types of events with markings, separately measure false alarms per hour and delay. The weight of the models and the source videos require a separate verification of the rights; the availability of the code does not confirm quality on the industrial stream.

<p align=center> Videoanalyst of TB violations at RCD stations. Computer vision module (AI)</p>
<hr>

## Contents
1. [Content](#Content)
2. [Description](#description)
3. [Installation (local)](#installation-local)
3. [Docker](# setup-docker)

## Description
The project is being conducted by the RZR station analyst. Resolves the following tasks:  
<table>
    <tr>
        <th>№</th>
        <th> Name </th>
    </tr>
    <tr>
        <td>1.1</td>
        <td> Identification of the Visibility Officer</td>
    </tr>
    <tr>
        <td>1.2</td>
        <td> Determination of the presence of outsiders in the designated area of the station</td>
    </tr>
    <tr>
        <td>1.3</td>
        <td> Determination of the location of a staff member without PPE in the designated area</td>
    </tr>
    <tr>
        <td>3.1</td>
        <td> Jumping from rolling stock, platform </td>
    </tr>
    <tr>
        <td>3.3</td>
        <td> Travel of staff member on foot, outside staircase of tank reciprocally incorporated with step</td>
    </tr>
    <tr>
        <td>3.4</td>
        <td> Train drafter's journey at the foot of the wagon in the designated area</td>
    </tr>
    <tr>
        <td>4.1</td>
        <td> Passing through the road in front of the wagons or entering the space between the wagons after starting to move</td>
    </tr>
    <tr>
        <td>4.2</td>
        <td> Railway crossing to moving rolling rolling stock</td>
    </tr>
    <tr>
        <td>6.1</td>
        <td> Work under standing wagons </td>
    </tr>
    <tr>
        <td>6.2</td>
        <td> Interception of workers on or under couplings </td>
    </tr>
    <tr>
        <td>7.1</td>
        <td> Identification of intersections of WD routes in prohibited areas</td>
    </tr>
    <tr>
        <td>7.2.1</td>
        <td>Identification of track crossings along a dangerous trajectory</td>
    </tr>
    <tr>
        <td>8.1</td>
        <td>Definition of workers sitting on rails, transformator boxes, other ground and ground devices</td>
    </tr>
    <tr>
        <td>9.1</td>
        <td> Definition of foreign objects in the designated area of the station</td>
    </tr>
    <tr>
        <td>10.1</td>
        <td> Definition of non-maintenance </td>
    </tr>
    <tr>
        <td>10.2</td>
        <td> Evidence of alcohol intoxication </td>
    </tr>

    
</table>

## Installation (local)
Locally required to install and deploy the project:
1. Make sure the computer has access to the video card (GPU)):
```shell
nvidia-smi
```  

Check the CUDA and cuDN versions:
```shell
nvcc --version
```
For the correct operation of the project ** it is recommended to install***CUDA v 12.4*.  
If the drivers are not installed, they must be installed!

2. Copy Project Files to Your Computer (through HTTPS or SSH)):
```shell
git clone https://github.com/Maxi-online/railway-video-event-analytics.git # HTTPS
git clone portfolio@example.org:ai-team/video_events-ai.git # SSH
```
3. downloading the Weights folder is the weight of models.
- Folder Reference Weights: [reference](https://example.org/models/model.pt) - Unload all model weights
  In video_events-ai/Weights folder.
4. Create virtual environment:
```shell
python -m venv .venv
```
5. Activate virtual environment:
```shell
.venv/Scripts/activate # Windows
source .venv/bin/activate # Linux
```
6. Set Dependency (torch + Torchvision + General)
```shell
# Torch + Torchvision supported by CUDA (GPU)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# Installation of General Dependences (accurate after installation of Torch))
pip install -r requirements.txt
```
7. Start the project:
```shell
# Start with default parameters (see spc/config/config.yaml))
python main.py

# Reset Start
python main.py cam_id=1 lwt_enabled=True r_enabled=True task_1_3_enabled=True task_3_1_enabled=True task_3_3_enabled=True task_4_2_enabled=True task_5_2_enabled=True task_6_1_enabled=True task_6_2_enabled=True task_7_enabled=True task_7_1_enabled=True task_8_1_1_enabled=True task_9_enabled=True task_10_1_enabled=True task_10_2_enabled=True db_enabled=True topic_name=cam_id_1 kafka_address=http://127.0.0.1:29092 kafka_enabled=False save_to_images_temp=False show_video_preview=False
```
8. To close the project:
- If show_video_preview=True, press ESC or q on the keyboard to close cv2 Windows (in position)).
- If show_video_preview=False, then to close the project - press Ctrl + C - to break from the keyboard (all flows will be stopped))


## Installation (Docker)
Docker is required to install and deploy the project:
P.P. 1-3 is similar to the installation (locally)). Look up: [instruction](#Installation-local)
4. Start the project with a team:
```shell
docker compose up -d --build # On first launch or if project code was updated
docker compose up-d # On subsequent launches
```

## Private configuration

Provider keys, tokens, passwords, proxy and personal parameters are set through variable environments. `.env.example` contains names without values. `.env` filled and local settings deleted from Git. Do not transmit secrets by command arguments and do not publish real magazines, downloads, certificates or user documents.

To run, export the required variables to the process environment or fill the ignored local `.env` if the entrypoint used is downloading it. The content of `.env` does not in itself become an environment of arbitrary Python command. For Docker Compose, the values of `.env` are transferred to customized services.

Hydra/OmegaConf reads secret fields from `src/config/config.yaml` via `${oc.env:NAME}`.

The names of the private parameters are given in [.env.example](.env.example)). Old values are only retained in the protected local storage facility of the owner; it is not part of the published project. Once the keys are rotated, use new values.

Checking the source and configuration formats does not replace the launch of the infrastructure and model evaluation. Privacy changes do not confirm the quality of AI, the performance of external APIs, or the willingness of the entire system to operate.

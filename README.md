# dms

## export dmd images

```bash
# activate the virtual environment
$ source Documents/anntool_py/bin/activate

# to the explore-material tool directory
$ cd Documents/project_dms/exploreMaterial-tool

# run the tool
$ python DExTool.py

# desitination directory
$ /home/han/Documents/project_dms
# dmd data directory (for type g)
$ /home/han/Documents/project_dms/dmd/gA
```

## usage

```bash
$ source Documents/anntool_py/bin/activate
$ cd Documents/DMD-Driver-Monitoring-Dataset/dms/facialTracking
$ python3 facialTracking.py
```

## all classes

```python
all_classes = [
    # 'driver_actions/change_gear',
    'driver_actions/drinking',
    'driver_actions/hair_and_makeup',
    'driver_actions/phonecall_left',
    'driver_actions/phonecall_right',
    # 'driver_actions/radio',
    'driver_actions/reach_backseat',
    'driver_actions/reach_side',
    'driver_actions/safe_drive',
    # 'driver_actions/standstill_or_waiting',
    'driver_actions/talking_to_passenger',
    'driver_actions/texting_left',
    'driver_actions/texting_right',
    # 'driver_actions/unclassified',
    'gaze_on_road/looking_road',
    # 'gaze_on_road/not_looking_road',
    # 'hand_on_gear/hand_on_gear',
    # 'hands_using_wheel/both',
    # 'hands_using_wheel/none',
    # 'hands_using_wheel/only_left',
    # 'hands_using_wheel/only_right',
    # 'objects_in_scene/bottle',
    # 'objects_in_scene/cellphone',
    # 'objects_in_scene/hair_comb',
    # 'talking/talking'
]
```

## plan

* detect face, eye status, gaze direction, yawn, emotion
```python
eye = ['open', 'closed']
gaze = ['left', 'right', 'middle']
mouth = ['closed', 'slight open', 'large open']
emotion = ['happy', 'neutral', 'unhappy']
```
* behavior detection using dmd
* dms to make decision
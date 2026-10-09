"""V7 Segmenter: a desktop application for the fine-tuned V7 PCB segmentation model.

Layers, each depending only on the ones above it in this list:

    domain     plain data: segment classes, image documents, predictions
    imaging    image I/O, overlay rendering and viewport maths (numpy/OpenCV only)
    inference  the model adapter around model_v7.py and the background worker
    services   application state, export, folder navigation, scratch workspace
    ui         tkinter views and the controller that connects them to services
    app        the composition root that builds and wires everything

model_v7.py is used read-only: it is imported as a module and its own inference
function, predict_one_image, produces every prediction.
"""

APP_NAME = "V7 Segmenter"
__version__ = "1.0.0"

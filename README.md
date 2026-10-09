# Model V7 Image Testing GUI

**V7 Segmenter** is a desktop application for testing the fine-tuned V7 PCB segmentation
model on your own images. Insert an image, press **Predict**, and the segmented objects are
drawn on top of it.

**Start it:** double-click `Start V7 GUI.bat`, or run `python run_gui.py [image]`. On a new computer, first run `Install requirements.bat` once (see Quick start).

## Using it

1. The window opens and the model loads in the background, which takes about 15 s.
   The status bar at the bottom right turns green when it is ready.
2. Click the **image icon** in the middle (or press Ctrl+O, or use File → Insert image) and
   choose a photo.
3. Press **Predict** (bottom right). The predicted segments are drawn over the photo in the
   colour of their class:

   | Colour | Class |
   |---|---|
   | red | Rectangle |
   | orange | Rectangle_concave |
   | green | circle |
   | blue | circle_full |

4. Press **Clear** (bottom right) to remove the photo and its prediction.

### Side panel

| Section | What it does |
|---|---|
| **Prediction** | Object count and time. Per class: a tick box to show or hide that class, and its count. |
| **Display** | Show or hide the segments, and set opacity and outlines. **View** switches to the model's other outputs: its own overlay with boxes and labels, the semantic class map, boundary probability, inner distance, or confidence. **Hide objects below confidence** filters what is drawn; the model's output itself is unchanged. |
| **Objects** | Every detected object with its class, confidence, area and centre. Click a column title to sort. Click a row, or an object on the photo, to highlight it. |

### Image view

| Action | How |
|---|---|
| Zoom | Mouse wheel, Ctrl + / Ctrl −, or the toolbar |
| Pan | Drag the photo |
| Fit to window | Ctrl+0 |
| Actual size | Ctrl+1 |
| Read a pixel | Hover. The status bar shows the pixel position and the object under the cursor. |

### More

- **Folder browsing:** PgUp / PgDn (or the arrow buttons) open the previous or next photo in
  the same folder. Tick **Predict automatically** to run the model on each photo as it opens.
- **Save overlay** (Ctrl+S): saves the photo with its segments at full resolution.
- **Export** (Ctrl+E): writes a folder containing:
  - the model's own outputs: `instances.json`, `instances.png`, `semantic_colour.png`,
    `boundary.png`, `inner_distance.png`, `semantic_confidence.png`, and the instance and
    class id maps (`.npy`)
  - `overlay.png`
  - `objects.csv`
  - `summary.json`
- **Recent images:** File → Recent images.
- **Help:** F1 lists the keyboard shortcuts; Help → About shows the model's details and scores.

## Quick start (from GitHub)

**You need:** Windows, Python 3.11, and Git. Git for Windows includes Git LFS.

```
git clone https://github.com/Jenit88/Model_V7_Image_Testing_GUI.git
cd Model_V7_Image_Testing_GUI
```

1. Double-click **`Install requirements.bat`**. This is needed once per computer.
2. Double-click **`Start V7 GUI.bat`**.

The model (`model_v7_resource/best_fine_tuned_model_v7_instance.keras`, 231 MB) is stored
with Git LFS, and `git clone` downloads it together with the code.

If the application says the model is only a *Git LFS placeholder*, Git LFS was not active
when you cloned. Fix it by running these commands in the folder:

```
git lfs install
git lfs pull
```

## The model

`model_v7_resource/` holds exactly what the application runs:

| File | What it is |
|---|---|
| `best_fine_tuned_model_v7_instance.keras` | The fine-tuned V7 model (epoch 54), copied unchanged; sha256 starts `872f9bc06ecf0a2e`. |
| `model_v7.py` | The V7 code, copied unchanged. The application imports it **read-only** (without writing anything into this folder). Every prediction is made by its own `predict_one_image`: letterbox to 512 × 512, model, instance decoding, confidence floor, and scaling back to the photo's own size. |
| `final_model_selection.json` | The training run's record of how this model was chosen (for reference only). |
| `training_logs/` | Every log of the training run that produced this model, copied unchanged (157 files): `training_log.csv` (69 epochs), `instance_checkpoint_history.json` (23 validation checks), `training_config.json`, `TRAINING_REPORT.md`, `model_summary.txt`, the final validation and test evaluations, TensorBoard event files and 140 validation previews. Not used by the application. |

| Data set | mAP50-95 |
|---|---|
| Validation, 3,128 images | 83.5% |
| New test set, 664 images | 71.5% |

TensorFlow on native Windows runs on the CPU, so one prediction takes about 2.5 s on this laptop.
The window stays responsive meanwhile.

## Architecture

```
GUI/
├─ Start V7 GUI.bat, run_gui.py       entry points (Install requirements.bat: one-time setup)
├─ config/
│  ├─ model_card.json                 model name, provenance and scores (shown in About)
│  └─ settings.json                   your preferences (written on exit)
├─ model_v7_resource/                 the model and model_v7.py, never modified
├─ logs/v7_segmenter.log              rotating log; Help → Open the log folder
├─ tests/                             unit tests, plus an optional model smoke test
└─ v7_segmenter/
   ├─ app.py            composition root: builds the services and the window, wires them
   ├─ paths.py, settings.py, logging_setup.py, events.py (publish/subscribe bus)
   ├─ domain/           plain data: SegmentClass, ImageDocument, DetectedObject, Prediction, ModelInfo
   ├─ imaging/          numpy/OpenCV only: io (read/write any Windows path),
   │                    render (overlay), viewport (zoom/pan maths)
   ├─ inference/        engine (adapter around model_v7.predict_one_image, the only TensorFlow user),
   │                    worker (one background thread; results return to the UI thread)
   ├─ services/         state (observable application state), export, navigation (folder stepping),
   │                    workspace (temporary folders for the model's outputs, removed on exit)
   ├─ ui/               main_window, image_view, side_panel, status_bar, dialogs, tooltip,
   │                    icons, theme, and controller (user actions → services → state)
   └─ demo.py           scripted end-to-end check of the real window
```

**Layers.** Each layer depends only on the layers before it: domain → imaging → inference /
services → ui → app.

**Data flow.** The views never change data themselves. A user action calls the controller.
The controller calls a service, or queues work for the background thread, and then updates
`AppState`. `AppState` publishes a topic on the event bus, and each view redraws what it
shows from the state.

## Tests

From this folder:

```
python -m unittest discover -s tests -t . -v
```

The model smoke test loads the real model and predicts one photo (about 20 s). It is opt-in:

```
set V7_MODEL_TESTS=1
set V7_TEST_IMAGE=C:\path\to\photo.png
python -m unittest tests.test_engine_smoke -v
```

The scripted check of the real window takes screenshots and leaves your settings untouched:

```
python run_gui.py --demo C:\path\to\photo.png --screenshots C:\temp\check
```

## Requirements

Python 3.11 with the packages in `requirements.txt`. They are already installed in this
laptop's Windows Python. tkinter comes with Python.

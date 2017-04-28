## Model development

For each estimator that requires a pre-trained model, the training scripts will be stored under `training/ESTIMATOR/` and have the following filename convention:

- `requirements.txt` : requirements file for training this model.  This should only be used for helper modules to facilitate training (eg, muda or pescador), and cannot be required for test-time prediction.
- `index_train.json` : a json file listing the identifiers (`track_id`) of training data. Must be parse-able into a `pandas` dataframe.
- `index_test.json`: like above, but for testing data.
- `README.md`: description of the model architecture, parameters, training strategy, etc.

- `00-setup.py` : any necessary preliminary processing.  This includes things like pre-computed data augmentation.
- `01-prepare.py` : pump construction, preliminary feature extraction
    - saves the `pump` as `resources/pump.pkl` (pickle)
- `02-train.py` : model construction and training.
    - loads `index_train.json`, working path, and `pump` object
    - saves the resulting model as `resources/model.h5`
    - tracks the version number of the model.
- `03-evaluate.py` : testing
    - loads the prebuilt model and test data index
    - cannot rely upon pre-computed features: testing must be end-to-end
    - calls the appropriate `mir_eval` function on each estimate
    - stores the resulting score array as `resources/test_data.json`

Other conventions:

- All training scripts should seed all random number generators.

- All models should maintain a version counter in `version.txt`, which is tied to the current git revision
  when the model is trained.  The version number is of the form `GIT-HASH.ITERATION`.

- All music/annotation data should live in an easily searched directory, with the convention
    - `directory/track_id.audio_ext`
    - `directory/track_id.jams_ext`
    - where `audio_ext` is the audio extension (eg, `ogg` or `flac`) and `jams_ext` is the annotation extension
      (`jams` or `jamz`).

- cached features should be locatable from the training index, using a similar convention to input data:
    - `feature_dir/track_id.h5` (or `npz`, etc.)

- the above goes for data augmentation as well, using the `track_id.augment_id.ext` convention.

- Common functionality and infrastructure for model training lives under `crema.utils`.

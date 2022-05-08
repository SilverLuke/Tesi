#!/usr/bin/env python
# coding: utf-8

# In[63]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[64]:


from sklearn.model_selection import train_test_split
from functools import *
from datetime import datetime
from datetime import timedelta
from time import time
from keras_tuner import Hyperband, BayesianOptimization, RandomSearch
from keras import callbacks as kc

from IRESNs_tensorflow.time_series_datasets import *
from IRESNs_tensorflow.models import *
from benchmarks import *
from general_hp import *

PROJECT_ROOT = os.path.abspath(os.getcwd() + os.sep + os.pardir)

DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets")
TB_ROOT = os.path.join(os.path.abspath(os.sep), "tmp", "tensorboard")

BENCHMARKS_ROOT = os.path.join(PROJECT_ROOT, "benchmarks")
WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, "plots", "weights")

print("       Data dir:", DATA_ROOT)
print(" Benchmarks dir:", BENCHMARKS_ROOT)
print("    Weights dir:", WEIGHTS_ROOT)
print("Tensorboard dir:", TB_ROOT)

MAX_EPOCHS = 5000  # Positive Integer. How many epochs the tuner train the model for each trials
PATIENCE = 10  # EarlyStopping
BENCHMARKS_TRIALS = 10  # How many times do the benchmark. 0 to skip BENCHMARKS
MS_VERBOSE = 0

hyperparameters = {
    'Spectral Radius': [0.1, 0.3, 0.5, 0.7, 0.9],
    'Input Scaling'  : [0.01, 0.1, 1.],
    'Bias Scaling'   : [0.01, 0.1, 1.],
    'Inter Scaling'  : [0.0, 0.01, 0.1, 1.],
}

SKIP = True  # Skip if a model is already tested?
OVERWRITE = False  # Redo the model selection for a model?

TUNER = "GridSearch"
TUNER_DESC = "Libras"

READOUT_ACTIVATION_BINARY = keras.activations.sigmoid
LOSS_FUNCTION_BINARY = keras.losses.BinaryCrossentropy()
READOUT_ACTIVATION = keras.activations.softmax  # https://www.tensorflow.org/api_docs/python/tf/keras/activations
LOSS_FUNCTION = keras.losses.SparseCategoricalCrossentropy()  # https://www.tensorflow.org/api_docs/python/tf/keras/losses

if not os.path.exists(TB_ROOT):
    os.makedirs(TB_ROOT)

TUNER_STRING = TUNER + "." + str(MAX_EPOCHS) + "mt." + TUNER_DESC
benchmarks = BenchmarksDB(load_path=os.path.join(BENCHMARKS_ROOT, TUNER_STRING + ".json"))


# In[65]:


def get_seed(names):
    dataset_name, class_name, experiment_name, model_name = names
    import hashlib
    union = dataset_name + class_name + experiment_name + model_name
    hashed = hashlib.md5(union.encode('UTF-8'))
    seed = int(hashed.hexdigest(), 16) % 4294967295  # limit to 32 bit length value
    return seed


def model_selection(build_model_fn, hyperparameters, dimensions,
                    names, train_set, val_set, verbosity=0):
    dataset_name, class_name, experiment_name, model_name = names
    x_train, y_train = train_set
    x_val, y_val = val_set
    tf.random.set_seed(get_seed(names))

    set_hps = {}

    if model_name == "ESN":
        set_hps['Spectral Radius'] = hyperparameters['Spectral Radius']
        set_hps['Input Scaling'] = hyperparameters['Input Scaling']
        set_hps['Bias Scaling'] = hyperparameters['Bias Scaling']
    else:
        if class_name == "Single SR Single Input Single Inter":
            set_hps['Spectral Radius'] = hyperparameters['Spectral Radius']
            set_hps['Input Scaling'] = hyperparameters['Input Scaling']
            set_hps['Inter Scaling'] = hyperparameters['Inter Scaling']
            set_hps['Bias Scaling'] = hyperparameters['Bias Scaling']
        elif class_name == "Multiple SR Single Input Single Inter":
            for i in range(dimensions):
                set_hps['Spectral Radius ' + str(i)] = hyperparameters['Spectral Radius']
            set_hps['Input Scaling'] = hyperparameters['Input Scaling']
            set_hps['Inter Scaling'] = hyperparameters['Inter Scaling']
            set_hps['Bias Scaling'] = hyperparameters['Bias Scaling']
        elif class_name == "Single SR Multiple Input Single Inter":
            set_hps['Spectral Radius'] = hyperparameters['Spectral Radius']
            for i in range(dimensions):
                set_hps['Input Scaling ' + str(i)] = hyperparameters['Input Scaling']
            set_hps['Inter Scaling'] = hyperparameters['Inter Scaling']
            set_hps['Bias Scaling'] = hyperparameters['Bias Scaling']
        elif class_name == "Single SR Single Input Multiple Inter":
            set_hps['Spectral Radius'] = hyperparameters['Spectral Radius']
            set_hps['Input Scaling'] = hyperparameters['Input Scaling']
            for i in range(dimensions):
                set_hps['Inter Scaling ' + str(i)] = hyperparameters['Inter Scaling']
            set_hps['Bias Scaling'] = hyperparameters['Bias Scaling']
        else:
            raise ValueError("Unknown Experiment class")

    indices = {key: 0 for key in set_hps.keys()}
    iterate = True
    iteration = 0
    score = [(0, 0)]
    best_hps = {}
    while iterate:
        hp2test = {}
        for key, vals in set_hps.items():
            hp2test[key] = vals[indices[key]]

        # logic
        iteration += 1

        model = build_model_fn(hp2test)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=MAX_EPOCHS,
                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                               restore_best_weights=True)],
                      verbose=verbosity)
        test_loss, accuracy = model.evaluate(x_val, y_val, verbose=verbosity)
        if accuracy > score[-1][0]:
            score.append((accuracy, iteration))
            best_hps = hp2test.copy()

        iterate = False
        for key, vals in set_hps.items():
            indices[key] += 1
            iterate = True
            if indices[key] % len(vals) == 0:
                indices[key] = 0
                iterate = False
            else:
                break

    iterations = 1
    for key, vals in set_hps.items():
        iterations *= len(vals)
    print("Expected iterations:", iterations, "Done:", iteration)
    return best_hps, score


def testing_model(build_model_fn, best_hps, names,
                  train_set, val_set, test_set,
                  tensorboard_path=None, benchmarks_verbose=0):
    dataset_name, class_name, experiment_name, model_name = names
    x_train, y_train = train_set
    x_val, y_val = val_set
    x_test, y_test = test_set

    # keras.callbacks.CallbackList([])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
    if tensorboard_path is not None:
        tensorboard_dir = tensorboard_path + model_name
        callbacks.append(keras.callbacks.TensorBoard(tensorboard_dir, profile_batch='500,500'))

    print("[{}] Running {} benchmarks".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), BENCHMARKS_TRIALS))

    test_model = None

    required_time = []
    train_acc = []
    val_acc = []
    test_acc = []

    tf.random.set_seed(get_seed(names))

    for i in range(BENCHMARKS_TRIALS):
        initial_time = time()

        test_model = build_model_fn(best_hps)
        history = test_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=MAX_EPOCHS,
                                 # perchè si usa il validation data?
                                 callbacks=callbacks, verbose=benchmarks_verbose)
        test_loss, accuracy = test_model.evaluate(x_test, y_test)

        required_time.append(time() - initial_time)

        train_acc.append(history.history['accuracy'][-1])
        val_acc.append(history.history['val_accuracy'][-1])
        test_acc.append(accuracy)

    stat = Statistic(train_acc, val_acc, test_acc, required_time, dict_hps=best_hps)

    return test_model, stat


import notify2  # TODO replace notify watch here : https://notify2.readthedocs.io/en/latest/

notify2_init = False


def send_notification(title, message):
    def notify2_init_fun():
        global notify2_init
        if not notify2_init:
            notify2.init("Tesi")

    notify2_init_fun()

    notice = notify2.Notification(title, message)
    notice.show()


# # Build model functions

# In[66]:


def build_ESN(output_classes, _reservoirs, hps) -> ESN:
    if output_classes == 2:
        output_units = 1
        readout_activation = READOUT_ACTIVATION_BINARY
        loss = LOSS_FUNCTION_BINARY
    else:
        output_units = output_classes
        readout_activation = READOUT_ACTIVATION
        loss = LOSS_FUNCTION

    tmp_model = ESN(units=100,
                    connectivity=1.,
                    spectral_radius=hps['Spectral Radius'],
                    input_scaling=hps['Input Scaling'],
                    bias_scaling=hps['Bias Scaling'],
                    leaky=0.01,
                    output_units=output_units,
                    readout_activation=readout_activation,
                    dtype=tf.float32
                    )

    alpha = 0.1
    tmp_model.compile(
        optimizer=keras.optimizers.Adam(alpha),
        loss=loss,
        metrics=['accuracy'],
    )
    return tmp_model


def get_normalization_iiresn(reservoirs, spectral_radius, inter_scaling):
    sr = None
    if isinstance(spectral_radius, float):
        sr = [spectral_radius for _ in range(reservoirs)]
    elif isinstance(spectral_radius, list):
        if len(spectral_radius) != reservoirs:
            raise IndexError
        sr = spectral_radius

    norm = None
    if isinstance(inter_scaling, float):
        norm = [[sr[i] if i == j else
                 inter_scaling
                 for i in range(reservoirs)]
                for j in range(reservoirs)]

    elif isinstance(inter_scaling, list):
        if len(inter_scaling) != reservoirs:
            raise IndexError
        norm = [[sr[i] if i == j else
                 inter_scaling[i]
                 for i in range(reservoirs)]
                for j in range(reservoirs)]
    return norm


def build_IIRESN(output_classes, reservoirs, hps) -> IIRESN:
    if output_classes == 2:
        output_units = 1
        readout_activation = READOUT_ACTIVATION_BINARY
        loss = LOSS_FUNCTION_BINARY
    else:
        output_units = output_classes
        readout_activation = READOUT_ACTIVATION
        loss = LOSS_FUNCTION

    try:
        spectral_radius = hps['Spectral Radius']
    except KeyError:
        spectral_radius = [hps['Spectral Radius ' + str(i)] for i in range(reservoirs)]

    try:
        input_scaling = hps['Input Scaling']
    except KeyError:
        input_scaling = [hps['Input Scaling ' + str(i)] for i in range(reservoirs)]

    try:
        inter_scaling = hps['Inter Scaling']
    except KeyError:
        inter_scaling = [hps['Inter Scaling ' + str(i)] for i in range(reservoirs)]


    tmp_model = IIRESN(units=100,
                       sub_reservoirs=reservoirs,
                       connectivity=[[1. for _ in range(reservoirs)] for _ in range(reservoirs)],
                       normalization=get_normalization_iiresn(reservoirs, spectral_radius, inter_scaling),
                       use_norm2=False,
                       input_scaling=input_scaling,
                       bias_scaling=hps['Bias Scaling'],
                       leaky=0.01,
                       gsr=None,
                       vsr=None,
                       output_units=output_units,
                       readout_activation=readout_activation,
                       dtype=tf.float32
                       )

    alpha = 0.1
    tmp_model.compile(
        optimizer=keras.optimizers.Adam(alpha),
        loss=loss,
        metrics=['accuracy'],
    )
    return tmp_model


# # Confiugrations
# 
# |        | ArticularyWordRecognition | CharacterTrajectories | Epilepsy | JapaneseVowels  | Libras | SpokenArabicDigits |
# |--------|:-------------------------:|:---------------------:|:--------:|:---------------:|:------:|:------------------:|
# | Input  |             9             |           3           |    3     |       12        |   2    |         13         |
# | Output |            25             |          20           |    4     |        9        |   15   |         10         |

# In[67]:



def get_name(fn):
    return fn.__annotations__['return'].__name__


config = {
    'Datasets'                             : [
        "Libras",
        #"Epilepsy",
        #"CharacterTrajectories",
        #"ArticularyWordRecognition",
        #"JapaneseVowels"
        #"SpokenArabicDigits",
    ],
    'Classes'                              : [
        'Reference',
        'Single SR Single Input Single Inter',
        'Multiple SR Single Input Single Inter',
        'Single SR Multiple Input Single Inter',
        'Single SR Single Input Multiple Inter',
    ],
    'Reference'                            : {
        'Models'     : [
            build_ESN
        ],
        'Experiments': {
            'Units 100': hyperparameters,
        }
    },
    'Single SR Single Input Single Inter'  : {
        'Models'     : [
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': hyperparameters,
        }
    },
    'Multiple SR Single Input Single Inter': {
        'Models'     : [
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': hyperparameters,
        }
    },
    'Single SR Multiple Input Single Inter': {
        'Models'     : [
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': hyperparameters,
        },
    },
    'Single SR Single Input Multiple Inter': {
        'Models'     : [
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': hyperparameters,
        },
    },
}


# # Run all

# In[68]:


def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


from plotting import plot_model

datasets = config.get('Datasets')
classes = config.get('Classes')

run_time = time()
for dataset_name in datasets:
    train_path = os.path.join(DATA_ROOT, dataset_name, dataset_name + '_TRAIN.ts')
    test_path = os.path.join(DATA_ROOT, dataset_name, dataset_name + '_TEST.ts')

    x_train_all, y_train_all = load_sktime_dataset(train_path)
    x_test, y_test = load_sktime_dataset(test_path)

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all,
                                                      test_size=0.33, random_state=42, stratify=y_train_all)

    train_set = (x_train.astype(np.float32), y_train)
    val_set = (x_val.astype(np.float32), y_val)
    test_set = (x_test.astype(np.float32), y_test)

    input_dim = x_train.shape[-1]
    output_units = len(np.unique(y_test))  # Dataset must have one of each features

    for class_name in classes:
        for experiment_name, hps in config.get(class_name).get("Experiments").items():
            for model_fn in config.get(class_name).get("Models"):
                model_name = get_name(model_fn)
                print("[{}] M.S. of {: >25} {: >37} {: >10} {: >10}".format(get_date(),
                                                                            dataset_name, class_name, experiment_name,
                                                                            model_name))
                already_tested = benchmarks.is_benchmarked(dataset_name, class_name, experiment_name, model_name)
                if already_tested and SKIP:
                    print("[                   ] Skip Already tested!")
                    continue
                start_model = time()
                build_fn = partial(model_fn, output_units, input_dim)
                names = (dataset_name, class_name, experiment_name, model_name)

                best_hp, score = model_selection(build_fn, hyperparameters, input_dim, names, train_set, val_set)

                duration = time() - start_model
                string_out = "[" + get_date() + "] M.S. run time " + str(timedelta(seconds=duration))
                print(string_out)

                if BENCHMARKS_TRIALS > 0:
                    model, stat = testing_model(build_fn, best_hp, names, train_set, val_set, test_set)
                    stat.add_score(score)
                    benchmarks.add(dataset_name, class_name, experiment_name, model_name, stat)
                    plot_model(model, names, path=WEIGHTS_ROOT, show=False)

                benchmarks.save()

duration = time() - run_time
string_out = "Requested time: " + str(timedelta(seconds=duration))
print(string_out)
send_notification("All Done", string_out)


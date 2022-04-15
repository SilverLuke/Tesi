#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[14]:


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
TUNER_ROOT = os.path.join(PROJECT_ROOT, "models")
TB_ROOT = os.path.join(os.path.abspath(os.sep), "tmp", "tensorboard")

BENCHMARKS_ROOT = os.path.join(PROJECT_ROOT, "benchmarks")
WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, "plots", "weights")

print("       Data dir:", DATA_ROOT)
print("      Tuner dir:", TUNER_ROOT)
print(" Benchmarks dir:", BENCHMARKS_ROOT)
print("    Weights dir:", WEIGHTS_ROOT)
print("Tensorboard dir:", TB_ROOT)

TUNER = "RandomSearch"  # "Hyperband" or "BayesianOptimization" or "RandomSearch"

CONFIGURATIONS = 500  # Positive Integer. How many configuration the tuner will try
MAX_EPOCHS = 500  # Positive Integer. How many epochs the tuner train the model for each trials
TRIALS = 1  # Positive Integer. How many iterations for one set of hyperparameters
PATIENCE = 10  # EarlyStopping
BENCHMARKS_TRIALS = 10  # How many times do the benchmark. 0 to skip BENCHMARKS
MS_VERBOSE = 0

MAX_UNITS = 250
UNITS = sorted([50, 75, 100, 150, 250])
MINVAL = 0.01  # Positive Float. How
MAXVAL = 1.5

SKIP = True  # Skip if a model is already tested?
OVERWRITE = False  # Redo the model selection for a model?

TUNER_DESC = "Esperimenti"

READOUT_ACTIVATION_BINARY = keras.activations.sigmoid
LOSS_FUNCTION_BINARY = keras.losses.BinaryCrossentropy()
READOUT_ACTIVATION = keras.activations.softmax  # https://www.tensorflow.org/api_docs/python/tf/keras/activations
LOSS_FUNCTION = keras.losses.SparseCategoricalCrossentropy()  # https://www.tensorflow.org/api_docs/python/tf/keras/losses

if not os.path.exists(TUNER_ROOT):
    os.makedirs(TUNER_ROOT)
if not os.path.exists(TB_ROOT):
    os.makedirs(TB_ROOT)

TUNER_STRING = TUNER + "." + str(MAX_EPOCHS) + "me" + str(CONFIGURATIONS) + "mt." + TUNER_DESC
benchmarks = BenchmarksDB(load_path=os.path.join(BENCHMARKS_ROOT, TUNER_STRING + ".json"))


# In[15]:


class CurrentMSModel(kc.Callback):
    def __init__(self, names):
        super().__init__()
        self.dataset_name, self.class_name, self.experiment_name, self.model_name = names

    def on_train_begin(self, logs=None):
        #ids.clear_output(wait=True)
        print("MS of {} {} {} {}".format(self.dataset_name, self.class_name, self.experiment_name, self.model_name))


def get_seed(names):
    dataset_name, class_name, experiment_name, model_name = names
    import hashlib
    union = dataset_name + class_name + experiment_name + model_name
    hashed = hashlib.md5(union.encode('UTF-8'))
    seed = int(hashed.hexdigest(), 16) % 4294967295  # limit to 32 bit length value
    return seed


def model_selection(build_model_fn, names,
                    train_set, val_set,
                    tuner_path, verbose=1):
    dataset_name, class_name, experiment_name, model_name = names
    x_train, y_train = train_set
    x_val, y_val = val_set
    seed = get_seed(names)
    if TUNER == "Hyperband":
        working_dir = os.path.join(tuner_path, TUNER_STRING, dataset_name, class_name)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        tuner = Hyperband(
            build_model_fn,
            objective='val_accuracy',
            max_epochs=MAX_EPOCHS,
            hyperband_iterations=1.,
            seed=seed,
            directory=working_dir,
            project_name=experiment_name + ' ' + model_name,
            overwrite=OVERWRITE,
            executions_per_trial=TRIALS,
        )
    elif TUNER == "BayesianOptimization":
        working_dir = os.path.join(tuner_path, TUNER_STRING, dataset_name, class_name)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        tuner = BayesianOptimization(
            build_model_fn,
            objective='val_accuracy',
            max_trials=CONFIGURATIONS,
            seed=seed,

            directory=working_dir,
            project_name=experiment_name + ' ' + model_name,
            overwrite=OVERWRITE,
            executions_per_trial=TRIALS,
        )
    elif TUNER == "RandomSearch":
        working_dir = os.path.join(tuner_path, TUNER_STRING, dataset_name, class_name)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        tuner = RandomSearch(
            build_model_fn,
            objective='val_accuracy',
            max_trials=CONFIGURATIONS,
            seed=seed,

            directory=working_dir,
            project_name=experiment_name + ' ' + model_name,
            overwrite=OVERWRITE,
            executions_per_trial=TRIALS,
        )
    else:
        raise ValueError("Unknown Tuner -> {}".format(TUNER))

    # now the tuner will search the best hyperparameters
    tuner.search(x_train, y_train, epochs=MAX_EPOCHS, validation_data=(x_val, y_val),
                 callbacks=[
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE),
                     #CurrentMSModel(names)
                 ], verbose=verbose)

    return tuner, tuner.oracle.get_best_trials(1)[0].score


def testing_model(names, tuner,
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

    best_model_hp = tuner.get_best_hyperparameters()[0]

    test_model = None

    required_time = []
    train_acc = []
    val_acc = []
    test_acc = []

    tf.random.set_seed(get_seed(names))

    for i in range(BENCHMARKS_TRIALS):
        initial_time = time()

        test_model = tuner.hypermodel.build(best_model_hp)
        history = test_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=MAX_EPOCHS,
                                 # perchè si usa il validation data?
                                 callbacks=callbacks, verbose=benchmarks_verbose)
        test_loss, accuracy = test_model.evaluate(x_test, y_test)

        required_time.append(time() - initial_time)

        train_acc.append(history.history['accuracy'][-1])
        val_acc.append(history.history['val_accuracy'][-1])
        test_acc.append(accuracy)

    stat = Statistic(best_model_hp, train_acc, val_acc, test_acc, required_time)

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

# In[16]:


def build_ESN(output_classes, _reservoirs,  # Defined by dataset
              hp, tuner) -> ESN:
    if output_classes == 2:
        output_units = 1
        readout_activation = READOUT_ACTIVATION_BINARY
        loss = LOSS_FUNCTION_BINARY
    else:
        output_units = output_classes
        readout_activation = READOUT_ACTIVATION
        loss = LOSS_FUNCTION

    tmp_model = ESN(units=hp.get_units(tuner, min_value=50, max_value=MAX_UNITS, choise=UNITS),
                    connectivity=hp.get_connectivity_esn(tuner),
                    spectral_radius=hp.get_normalization_esn(tuner, min_value=MINVAL, max_value=MAXVAL,
                                                             sampling="linear"),
                    input_scaling=hp.get_input_scaling(tuner, 1, min_value=MINVAL, max_value=MAXVAL, sampling="linear"),
                    bias_scaling=hp.get_bias_scaling(tuner, 1, min_value=MINVAL, max_value=MAXVAL, sampling="linear"),
                    leaky=hp.get_leaky(tuner),
                    output_units=output_units,
                    readout_activation=readout_activation,
                    dtype=tf.float32

                    )

    alpha = hp.get_learning_rate(tuner)
    tmp_model.compile(
        optimizer=keras.optimizers.Adam(alpha),
        loss=loss,
        metrics=['accuracy'],
    )
    return tmp_model


def build_IRESN(output_classes, reservoirs,  # Defined by dataset
                hp, tuner) -> IRESN:
    if output_classes == 2:
        output_units = 1
        readout_activation = READOUT_ACTIVATION_BINARY
        loss = LOSS_FUNCTION_BINARY
    else:
        output_units = output_classes
        readout_activation = READOUT_ACTIVATION
        loss = LOSS_FUNCTION

    tmp_model = IRESN(units=hp.get_units(tuner, min_value=50, max_value=MAX_UNITS, choise=UNITS),
                      sub_reservoirs=reservoirs,
                      connectivity=hp.get_connectivity_iresn(tuner, reservoirs),
                      normalization=hp.get_normalization_iresn(tuner, reservoirs, min_value=MINVAL,
                                                               max_value=MAXVAL),
                      input_scaling=hp.get_input_scaling(tuner, reservoirs, min_value=MINVAL,
                                                         max_value=MAXVAL, sampling="linear"),
                      bias_scaling=hp.get_bias_scaling(tuner, reservoirs, min_value=MINVAL, max_value=MAXVAL,
                                                       sampling="linear"),
                      leaky=hp.get_leaky(tuner),
                      gsr=hp.get_gsr(tuner, min_value=MINVAL, max_value=MAXVAL),
                      vsr=hp.get_vsr(tuner, reservoirs),
                      output_units=output_units,
                      readout_activation=readout_activation,
                      dtype=tf.float32
                      )

    alpha = hp.get_learning_rate(tuner)
    tmp_model.compile(
        optimizer=keras.optimizers.Adam(alpha),  # keras.optimizers.RMSprop(alpha),
        loss=loss,
        metrics=['accuracy'],
    )
    return tmp_model


def build_IIRESN(output_classes, reservoirs,  # Defined by dataset
                 hp, tuner) -> IIRESN:
    if output_classes == 2:
        output_units = 1
        readout_activation = READOUT_ACTIVATION_BINARY
        loss = LOSS_FUNCTION_BINARY
    else:
        output_units = output_classes
        readout_activation = READOUT_ACTIVATION
        loss = LOSS_FUNCTION

    tmp_model = IIRESN(units=hp.get_units(tuner, min_value=50, max_value=MAX_UNITS, choise=UNITS),
                       sub_reservoirs=reservoirs,
                       connectivity=hp.get_connectivity_iiresn(tuner, reservoirs),
                       normalization=hp.get_normalization_iiresn(tuner, reservoirs),
                       use_norm2=hp.use_norm2,
                       input_scaling=hp.get_input_scaling(tuner, reservoirs, min_value=MINVAL,
                                                          max_value=MAXVAL, sampling="linear"),
                       bias_scaling=hp.get_bias_scaling(tuner, reservoirs, min_value=MINVAL,
                                                        max_value=MAXVAL, sampling="linear"),
                       leaky=hp.get_leaky(tuner),
                       gsr=hp.get_gsr(tuner, min_value=MINVAL, max_value=MAXVAL),
                       vsr=hp.get_vsr(tuner, reservoirs),
                       output_units=output_units,
                       readout_activation=readout_activation,
                       dtype=tf.float32
                       )

    alpha = hp.get_learning_rate(tuner)
    tmp_model.compile(
        optimizer=keras.optimizers.Adam(alpha),  # keras.optimizers.RMSprop(alpha),
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

# In[17]:



def get_name(fn):
    return fn.__annotations__['return'].__name__


from general_hp import HP, HP_Manager

single_sr = HP_Manager(units=HP.fixed(100),
                       norm_sub_reservoirs=HP.restricted(),
                       norm_inter_connectivity=HP.restricted(),
                       connectivity_subr=HP.fixed(1.),
                       connectivity_inter=HP.restricted(),
                       input_scaling=HP.restricted(),
                       bias_scaling=HP.restricted(),
                       gsr=HP.fixed(False),
                       vsr=HP.fixed(False),
                       )
single_sr_vsr = HP_Manager(units=HP.fixed(100),
                           norm_sub_reservoirs=HP.restricted(),
                           norm_inter_connectivity=HP.restricted(),
                           connectivity_subr=HP.fixed(1.),
                           connectivity_inter=HP.restricted(),
                           input_scaling=HP.restricted(),
                           bias_scaling=HP.restricted(),
                           gsr=HP.fixed(False),
                           vsr=HP.fixed(True),
                           )
multiple_sr = HP_Manager(units=HP.fixed(100),
                         norm_sub_reservoirs=HP.free(),
                         norm_inter_connectivity=HP.restricted(),
                         connectivity_subr=HP.fixed(1.),
                         connectivity_inter=HP.restricted(),
                         input_scaling=HP.restricted(),
                         bias_scaling=HP.restricted(),
                         gsr=HP.fixed(False),
                         vsr=HP.fixed(False),
                         )
multiple_sr_vsr = HP_Manager(units=HP.fixed(100),
                             norm_sub_reservoirs=HP.free(),
                             norm_inter_connectivity=HP.restricted(),
                             connectivity_subr=HP.fixed(1.),
                             connectivity_inter=HP.restricted(),
                             input_scaling=HP.restricted(),
                             bias_scaling=HP.restricted(),
                             gsr=HP.fixed(False),
                             vsr=HP.fixed(True),
                             )
multiple_sr_multiple_is = HP_Manager(units=HP.fixed(100),
                                     norm_sub_reservoirs=HP.free(),
                                     norm_inter_connectivity=HP.restricted(),
                                     connectivity_subr=HP.fixed(1.),
                                     connectivity_inter=HP.restricted(),
                                     input_scaling=HP.free(),
                                     bias_scaling=HP.restricted(),
                                     gsr=HP.fixed(False),
                                     vsr=HP.fixed(False),
                                     )
multiple_sr_multiple_is_vsr = HP_Manager(units=HP.fixed(100),
                                         norm_sub_reservoirs=HP.free(),
                                         norm_inter_connectivity=HP.restricted(),
                                         connectivity_subr=HP.fixed(1.),
                                         connectivity_inter=HP.restricted(),
                                         input_scaling=HP.free(),
                                         bias_scaling=HP.restricted(),
                                         gsr=HP.fixed(False),
                                         vsr=HP.fixed(True),
                                         )

config = {
    'Datasets'                   : [
        #"CharacterTrajectories",
        #"Libras",
        #"SpokenArabicDigits",
        "ArticularyWordRecognition",
        #"Epilepsy",
        #"JapaneseVowels"
    ],
    'Classes'                    : [
        #'Reference',
        #'Single SR Single IS',
        #'Single SR Single IS VSR',
        #'Multiple SR Single IS',
        #'Multiple SR Single IS VSR',
        'Multiple SR Multiple IS',
        'Multiple SR Multiple IS VSR',
    ],
    'Reference'                  : {
        'Models'     : [
            build_ESN
        ],
        'Experiments': {
            'Units 100': single_sr
        }
    },
    'Single SR'                  : {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': single_sr,
        }
    },
    'Single SR vsr'              : {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': single_sr_vsr,
        }
    },
    'Multiple SR'                : {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': multiple_sr
        },
    },
    'Multiple SR vsr'            : {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': multiple_sr_vsr
        },
    },
    'Multiple SR Multiple IS'    : {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': multiple_sr_multiple_is
        },
    },
    'Multiple SR Multiple IS vsr': {
        'Models'     : [
            build_IRESN,
            build_IIRESN,
        ],
        'Experiments': {
            'Units 100': multiple_sr_multiple_is_vsr
        },
    },
}


# # Run all

# In[18]:


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

    features = x_train.shape[-1]
    output_units = len(np.unique(y_test))  # Dataset must have one of each features

    for class_name in classes:
        for experiment_name, hps in config.get(class_name).get("Experiments").items():
            for model_fn in config.get(class_name).get("Models"):
                model_name = get_name(model_fn)
                print("[{}] M.S. of {: >25} {: >15} {: >10} {: >10}".
                      format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                             dataset_name, class_name, experiment_name, model_name))
                already_tested = benchmarks.is_benchmarked(dataset_name, class_name, experiment_name, model_name)
                if already_tested and SKIP:
                    print("[                   ] Skip Already tested!")
                    continue
                start_model = time()
                build_fn = partial(model_fn, output_units, features, hps)
                names = (dataset_name, class_name, experiment_name, model_name)

                tuner, score = model_selection(build_fn, names,
                                               train_set, val_set,
                                               tuner_path=TUNER_ROOT, verbose=MS_VERBOSE)

                duration = time() - start_model
                string_out = "[" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "] M.S. run time " + str(
                    timedelta(seconds=duration))
                print(string_out)

                if BENCHMARKS_TRIALS > 0:
                    model, stat = testing_model(names, tuner, train_set, val_set, test_set)
                    stat.add_score(score)
                    benchmarks.add(dataset_name, class_name, experiment_name, model_name, stat)
                    plot_model(model, names, path=WEIGHTS_ROOT, show=False)

                benchmarks.save()
    benchmarks.save()
duration = time() - run_time
string_out = "Requested time: " + str(timedelta(seconds=duration))
print(string_out)
send_notification("All Done", string_out)


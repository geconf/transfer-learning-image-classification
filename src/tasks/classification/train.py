import argparse
import os
import datetime
import yaml
import shutil
import __init__ as booger

from tasks.classification.modules.trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. This should be the path to the the folder that contains all subtypes',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=False,
        default='config/arch/model.yaml',
        help='Architecture yaml config file. See /config/arch for an example.'
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='config/labels/data.yaml',
        help='Data yaml config file. See /config/labels for an example'
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
        datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put log data. Default: ~/logs/date+time'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # Print argument summary
    print("----------")
    print("SETTINGS")
    print("dataset", FLAGS.dataset)
    print("arch_cfg", FLAGS.arch_cfg)
    print("data_cfg", FLAGS.data_cfg)
    print("log", FLAGS.log)
    print("----------\n")

    # open arch config file
    try:
        print("Opening architecture config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening architecture yaml file. Exiting...")
        quit()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file. Exiting...")
    
    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory, check permissions. Exiting...")
        quit()
    
    # copy files to log folder (to record how the model was trained)
    # this is also used when opening the model during inference
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        shutil.copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
        shutil.copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    # create trainer and start the training
    trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log)
    #trainer.train()
    
    
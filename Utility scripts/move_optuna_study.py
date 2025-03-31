'''
This script moves a study from one storage to another. 
This means it copies the study to the new storage and deletes the study from the old storage.
If the new storage does not exist, it creates a new storage.

Usage:
    python move_optuna_study.py --old_storage sqlite:///old.db --new_storage sqlite:///new.db --study_name study_name
'''

import optuna
import argparse

def move_study(old_storage, new_storage, study_name):
    optuna.copy_study(from_study_name=study_name, from_storage=old_storage, to_storage=new_storage)
    optuna.delete_study(study_name=study_name, storage=old_storage)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_storage', type=str, required=True, help='Old storage URL')
    parser.add_argument('--new_storage', type=str, required=True, help='New storage URL')
    parser.add_argument('--study_name', type=str, required=True, help='Study name')
    args = parser.parse_args()

    move_study(args.old_storage, args.new_storage, args.study_name)
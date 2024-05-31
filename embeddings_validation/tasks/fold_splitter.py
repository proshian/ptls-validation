import os

import luigi
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from embeddings_validation.config import Config
from embeddings_validation.file_reader import TargetFile, IdFile


class FoldSplitter(luigi.Task):
    """
    Splits data into n folds (train, validation, test),
    saves them to files and creates folds.json file with paths to these files.

    The saved files are dumped TargetFile objects that contain 
    ids and target values (subset of the original data).
    Use TargetFile.load(path) to load theese files.
    TargetFile = embeddings_validation.file_reader.TargetFile

    Main interface of TargetFile are pseudo properties:
    * .ids_values
    * .target_values


    The Target (Luigi Target) of this task is folds.json file with folds information.

    The split can be done in two ways (self.conf.validation_schema):
    * VALID_TRAIN_TEST
    * VALID_CROSS_VAL

    For both split types the folds Dict has the same structure:
    Keys represent fold number and values are dictionaries with keys:
    - 'train': dictionary with keys 'path' and 'shape' representing path to the train data and its shape
    - 'valid': dictionary with keys 'path' and 'shape' representing path to the validation data and its shape
    - 'test': dictionary with keys 'path' and 'shape' representing path to the test data and its shape

    
    Test data is optional. If it is not provided, the 'test' property is None.
    If it's providded, regardless of the split type, the 'test' 
    property is always the same (same path and shape for all folds).

    If validation_schema is VALID_CROSS_VAL:
    * Fold numbers are integers from 0 to self.conf['split']['cv_split_count'] - 1
    * Each fold contains train and validation data randomly split from the train data

    If validation_schema is VALID_TRAIN_TEST:
    * Fold numbers are integers from 0 to self.conf['split']['n_iteration'] - 1
    * Each fold is exactly the same (Uses given train, validation and test ids); 
        same path and shape for all folds
    """
    
    conf = luigi.Parameter()

    _split_info = {}

    def output(self):
        path = os.path.join(self.conf.work_dir, 'folds', 'folds.json')
        return luigi.LocalTarget(path)

    def run(self):
        validation_schema = self.conf.validation_schema

        self.output().makedirs()

        folds = None
        if validation_schema == Config.VALID_TRAIN_TEST:
            folds = self.train_test_split()
        if validation_schema == Config.VALID_CROSS_VAL:
            folds = self.cross_val_split()
        assert folds is not None

        folds['_split_info'] = self._split_info
        with self.output().open('w') as f:
            json.dump(folds, f, indent=2)

    def _read_id_split_save(self, df_target, ids, save_path):
        df_target_fold = df_target.select_ids(ids)
        path = os.path.join(self.conf.work_dir, 'folds', save_path)
        df_target_fold = self.shuffle_before_dump(df_target_fold)
        df_target_fold.dump(path)
        return {'path': path, 'shape': df_target_fold.df.shape}

    def _select_pos_save(self, df_target, pos, save_path):
        df_target_fold = df_target.select_pos(pos)
        path = os.path.join(self.conf.work_dir, 'folds', save_path)
        df_target_fold = self.shuffle_before_dump(df_target_fold)
        df_target_fold.dump(path)
        return {'path': path, 'shape': df_target_fold.df.shape}

    def shuffle_before_dump(self, df_target_fold):
        row_order_shuffle_seed = self.conf['split'].get('row_order_shuffle_seed', None)
        if row_order_shuffle_seed is None:
            return df_target_fold

        new_target_fold = df_target_fold.clone_schema()
        shuffle_ix = np.random.choice(len(df_target_fold), len(df_target_fold), replace=False)
        df = df_target_fold.df
        df = df.sort_values(df_target_fold.cols_id)
        df = df.iloc[shuffle_ix].reset_index(drop=True)
        new_target_fold.df = df
        return new_target_fold

    def train_test_split(self):
        df_target = TargetFile.read_table(self.conf, **self.conf['target'])
        ids_train = IdFile.read_table(self.conf, **self.conf['split']['train_id'])
        ids_valid = IdFile.read_table(self.conf, **self.conf['split']['valid_id'])
        if 'test_id' in self.conf['split']:
            ids_test = IdFile.read_table(self.conf, **self.conf['split']['test_id'])
        else:
            ids_test = None
        ids_train, ids_valid, ids_test = self.check_ids_intersection(ids_train, ids_valid, ids_test)

        train_info = self._read_id_split_save(df_target, ids_train, 'target_train.pickle')
        valid_info = self._read_id_split_save(df_target, ids_valid, 'target_valid.pickle')
        if 'test_id' in self.conf['split']:
            test_info = self._read_id_split_save(df_target, ids_test, 'target_test.pickle')
        else:
            test_info = None

        folds = {
            i: {
                'train': train_info,
                'valid': valid_info,
                'test': test_info,
            } for i in range(self.conf['split']['n_iteration'])
        }
        return folds

    def get_folds(self, df_target):
        cv_split_count = self.conf['split']['cv_split_count']
        random_state = self.conf['split']['random_state']
        if self.conf['split']['is_stratify']:
            skf = StratifiedKFold(cv_split_count, shuffle=True, random_state=random_state)
            return skf.split(df_target.df, df_target.target_values)
        else:
            sf = KFold(cv_split_count, shuffle=True, random_state=random_state)
            return sf.split(df_target.df)

    def cross_val_split(self):

        df_target = TargetFile.read_table(self.conf, **self.conf['target'])

        if 'test_id' in self.conf['split']:
            ids_test = IdFile.read_table(self.conf, **self.conf['split']['test_id'])
        else:
            ids_test = None
        ids_train = IdFile.read_table(self.conf, **self.conf['split']['train_id'])
        ids_train, _, ids_test = self.check_ids_intersection(ids_train, None, ids_test)

        if 'test_id' in self.conf['split']:
            test_info = self._read_id_split_save(df_target, ids_test, 'target_test.pickle')
        else:
            test_info = None

        df_target_train = df_target.select_ids(ids_train)

        folds = {}
        for i, (i_train, i_valid) in enumerate(self.get_folds(df_target_train)):
            train_info = self._select_pos_save(df_target_train, i_train, f'target_train_{i}.pickle')
            valid_info = self._select_pos_save(df_target_train, i_valid, f'target_valid_{i}.pickle')

            folds[i] = {
                'train': train_info,
                'valid': valid_info,
                'test': test_info,
            }

        return folds

    def check_ids_intersection(self, train, valid, test):
        """

        :param train: mandatory
        :param valid: depends on validation schema
        :param test: optional
        :return:
        """
        def check_intersect(df1, df2, pair_name):
            """

            :param df1: not changed
            :param df2: records from df1 wil be removed
            :param pair_name:
            :return:
            """
            if df1 is None or df2 is None:
                return df1, df2

            if self.conf['split']['fit_ids']:
                df2, excluded_cnt = df2.exclude_ids(df1)
                if len(df2) == 0:
                    raise AttributeError(f'All records excluded in {pair_name}')
                self._split_info['check_intersect'][pair_name] = {'excluded_cnt': excluded_cnt}

            else:
                df = df2.select_ids(df1)
                if len(df) > 0:
                    raise AttributeError(f'Found id intersection in {pair_name}. '
                                         f'{len(df)} ({len(df) / len(df2) * 100}%) common records:\n'
                                         f'{df.df.head(2)}')
                self._split_info['check_intersect'][pair_name] = {'check': 'done'}
            return df1, df2

        if train is None:
            raise AssertionError(f'Incorrect splits: {train is not None}, {valid is not None}, {test is not None}')
        self._split_info['check_intersect'] = {}
        self._split_info['check_intersect']['fit_ids'] = self.conf['split']['fit_ids']

        test, valid = check_intersect(test, valid, 'test-valid')
        test, train = check_intersect(test, train, 'test-train')
        valid, train = check_intersect(valid, train, 'valid-train')

        return train, valid, test

import os
import numpy as np
from deap import creator
from dask import multiprocessing
from Code.utils import load_split_dataset
from tpot import TPOTClassifier, TPOTRegressor


#  prevent from TPOT to crash
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('forkserver')
    except RuntimeError as RTE:
        print(str(RTE))


class PipelineEvaluator:
    __doc__ = """Tool to evaluate pipeline scores at Given Dataset."""

    def __init__(self, dataset_path, json_path, n_jobs=1, config_dict=None, task="Classification"):
        self.scores = []
        self.datasets_path = dataset_path

        self.JSON = json_path

        if task == "Classification":
            self.tpot = TPOTClassifier(population_size=1, generations=0, verbosity=0,
                                       n_jobs=n_jobs, config_dict=config_dict, warm_start=True)
        elif task == "Regression":
            self.tpot = TPOTRegressor(population_size=1, generations=0, verbosity=0,
                                       n_jobs=n_jobs, config_dict=config_dict, warm_start=True)
        else:
            raise ValueError

        self.tpot._fit_init()  # Create _pset(PrimitiveSet)

    def evaluate(self, dataset_name, pipeline_list):
        """ Evaluate each pipeline of the given  List and save split datasets at the given path.

        :param pipeline_list: List of Tuples, first index will be dataSet namefor example 'MNIST'
        and the second index will be Evaluate Pipeline number for example '1984'.
        :param split_datasets_save_path: Sting. Path to destination directory.
        :param train_test_split_size: Double. Double represent the test\train split ratio.
        :return: evaluated_individuals_, scores. Dictionary, Dictionary.
                    Dictionary, key will be full pipeline as a string, value will
                            be the pipeline score according to predict test result.
                    Dictionary, key is predict of test result, value will
                            be the pipeline as Individual.
        """
        if type(pipeline_list) is not list:
            raise Exception("File not list")

        #  -------------- DATASET --------------
        X_train, y_train, X_test, y_test = load_split_dataset(self.datasets_path,dataset_name)
        pop = []

        for pipeline in pipeline_list:
            # Search the origin full pipeline
            JSONDict = np.load(os.path.join(self.JSON, f'{pipeline[0]}.npy'), allow_pickle=True).item()
            pipeline_string = JSONDict['Evaluate Pipeline ' + str(int(pipeline[1])) + ':']['TEST PARSING PIPELINE']

            # Create Individual object for Population List
            deap_pipeline = creator.Individual.from_string(pipeline_string, self.tpot._pset)
            pop.append(deap_pipeline)

        # Update tpot Object fields
        self.tpot.population_size = len(pop)
        self.tpot._pop = pop
        self.tpot.fit(X_train, y_train)
        for ind in pop:
            try:
                self.tpot._optimized_pipeline = ind
                self.tpot._summary_of_best_pipeline(X_train, y_train)
                ind_score = self.tpot.score(X_test, y_test)
                self.scores.append((ind_score, ind))
            except Exception as e:
                self.scores.append((np.NaN, ind))

        return self.tpot.evaluated_individuals_, self.scores

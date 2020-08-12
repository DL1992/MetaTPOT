import random
import warnings
from deap import gp
from Code.MetaTPOT_utils import *
from functools import partial
from datetime import datetime
from tpot import TPOTClassifier
from multiprocessing import cpu_count
from tpot.config import regressor_config_dict
from tpot.gp_deap import _wrapped_cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import Parallel, delayed

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class MetaTPOTClassifier(TPOTClassifier):
    """
    Main Class - implements MetaTPOTClassifier
    """

    def __init__(self, model, dataset_meta_features_df, primitive_to_hash_dic, le, knowledge_base_path, json_path,
                 longest_pipeline_size, rank_size=100, use_meta_model_flag=True, boosting=False, boosting_percent=0.1,
                 meta_selection_window_size=10, meta_selection=False, meta_selection_type='offspring',
                 meta_fitness_flag=False,
                 **kwargs):
        self.meta_fitness_flag = meta_fitness_flag
        self.use_boosting_flag = boosting
        super().__init__(**kwargs)
        self.knowledge_base_path = knowledge_base_path
        self.json_path = json_path
        if self.use_boosting_flag and (self.knowledge_base_path is None or self.json_path is None):
            raise ValueError("Cannot use boosting without knowledge-base or Json path")
        self.meta_features_df = dataset_meta_features_df
        self.longest_pipeline_size = longest_pipeline_size
        self.use_metamodel_flag = use_meta_model_flag
        self.meta_selection_type = meta_selection_type
        self.boosting_size = int(self.population_size * boosting_percent)
        self.meta_selection_size = meta_selection_window_size
        self.meta_selection_flag = meta_selection
        self.rank_size = rank_size
        self.meta_model = model
        self.primitives_to_hash_dic = primitive_to_hash_dic
        self.gptree = gp.PrimitiveTree(list())
        self.le = le
        self.gen_results_dic = {}

    def _meta_evaluate_individuals(self, population, features, target, sample_weight=None, groups=None):
        test_df = create_ranking_df_from_pop(population, self.primitives_to_hash_dic, self.gptree, self._pset,
                                             self.meta_features_df, self.longest_pipeline_size)
        top_pop_index = rank_pop(self.meta_model, test_df, len(population), self.le)
        top_pop = [population[i] for i in top_pop_index]
        population = top_pop
        # Evaluate the individuals with an invalid fitness
        individuals = [ind for ind in population if not ind.fitness.valid]
        # update pbar for valid individuals (with fitness values)
        if self.verbosity > 0:
            self._pbar.update(len(population) - len(individuals))

        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = self._preprocess_individuals(
            individuals)

        # Make the partial function that will be called below
        partial_wrapped_cross_val_score = partial(
            _wrapped_cross_val_score,
            features=features,
            target=target,
            cv=self.cv,
            scoring_function=self.scoring_function,
            sample_weight=sample_weight,
            groups=groups,
            timeout=max(int(self.max_eval_time_mins * 60), 1),
            use_dask=self.use_dask
        )

        result_score_list = []
        try:
            # Don't use parallelization if n_jobs==1
            if self._n_jobs == 1 and not self.use_dask:
                for sklearn_pipeline in sklearn_pipeline_list:
                    self._stop_by_max_time_mins()
                    val = partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)
                    result_score_list = self._update_val(val, result_score_list)
            else:
                # chunk size for pbar update
                if self.use_dask:
                    # chunk size is min of _lambda and n_jobs * 10
                    chunk_size = min(self._lambda, self._n_jobs * 10)
                else:
                    # chunk size is min of cpu_count * 2 and n_jobs * 4
                    chunk_size = min(cpu_count() * 2, self._n_jobs * 4)
                for chunk_idx in range(0, len(sklearn_pipeline_list), chunk_size):
                    self._stop_by_max_time_mins()
                    if self.use_dask:
                        import dask
                        tmp_result_scores = [
                            partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)
                            for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx + chunk_size]
                        ]

                        self.dask_graphs_ = tmp_result_scores
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            tmp_result_scores = list(dask.compute(*tmp_result_scores))

                    else:

                        parallel = Parallel(n_jobs=self._n_jobs, verbose=0, pre_dispatch='2*n_jobs')
                        tmp_result_scores = parallel(
                            delayed(partial_wrapped_cross_val_score)(sklearn_pipeline=sklearn_pipeline)
                            for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx + chunk_size])
                    # update pbar
                    for val in tmp_result_scores:
                        result_score_list = self._update_val(val, result_score_list)

        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write('', file=self._file)
                self._pbar.write('{}\nTPOT closed during evaluation in one generation.\n'
                                 'WARNING: TPOT may not provide a good pipeline if TPOT is stopped/interrupted in a early generation.'.format(
                    e),
                    file=self._file)
            # number of individuals already evaluated in this generation
            num_eval_ind = len(result_score_list)
            for i, val in enumerate(result_score_list):
                result_score_list[i] = 0.8 * val + 0.2 * (1 / (i + 1))
            self._update_evaluated_individuals_(result_score_list,
                                                eval_individuals_str[:num_eval_ind],
                                                operator_counts,
                                                stats_dicts)
            for ind in individuals[:num_eval_ind]:
                ind_str = str(ind)
                ind.fitness.values = (self.evaluated_individuals_[ind_str]['operator_count'],
                                      self.evaluated_individuals_[ind_str]['internal_cv_score'])
            # for individuals were not evaluated in this generation, TPOT will assign a bad fitness score
            for ind in individuals[num_eval_ind:]:
                ind.fitness.values = (5000., -float('inf'))

            self._pareto_front.update(population)

            self._pop = population
            raise KeyboardInterrupt

        for i, val in enumerate(result_score_list):
            result_score_list[i] = 0.8 * val + 0.2 * (1 / (i + 1))
        self._update_evaluated_individuals_(result_score_list, eval_individuals_str, operator_counts, stats_dicts)

        for ind in individuals:
            ind_str = str(ind)
            ind.fitness.values = (self.evaluated_individuals_[ind_str]['operator_count'],
                                  self.evaluated_individuals_[ind_str]['internal_cv_score'])
        individuals = [ind for ind in population if not ind.fitness.valid]
        self._pareto_front.update(population)

        return population

    def _boost_population(self):
        knowledge_base_df = pd.read_csv(self.knowledge_base_path)
        pipelines_loc_df = knowledge_base_df[['dataset', 'pipeline_num_in_dataset']].copy()
        test_df = create_ranking_df(knowledge_base_df, self.meta_features_df, self.longest_pipeline_size)
        top_pipelines = rank_pop(self.meta_model, test_df, self.boosting_size, self.le)
        pipelines_loc_df = pipelines_loc_df.iloc[top_pipelines]
        top_pipelines = [tuple(x) for x in pipelines_loc_df.values]
        boost_pipelines = get_TPOT_pipelines(top_pipelines, self.json_path, self._pset)
        return boost_pipelines

    def _test_gen(self, gen=0, pop=None):
        self._check_periodic_pipeline(gen=gen)
        pop_scores = [ind.fitness.wvalues[1] for ind in pop]
        pop_scores = np.array(pop_scores)
        gen_avg_score = np.nanmean(pop_scores)
        gen_median_score = np.nanmedian(pop_scores)
        gen_max_score = np.nanmax(pop_scores)
        gen_min_score = np.nanmin(pop_scores)
        self.gen_results_dic[gen] = [gen, gen_avg_score, gen_median_score, gen_max_score, gen_min_score,
                                     pop_scores.size]

    def fit(self, features, target, sample_weight=None, groups=None):
        """Fit an optimized machine learning pipeline.
                Uses genetic programming to optimize a machine learning pipeline that
                maximizes score on the provided features and target. Performs internal
                k-fold cross-validaton to avoid overfitting on the provided data. The
                best pipeline is then trained on the entire set of provided samples.
                Parameters
                ----------
                features: array-like {n_samples, n_features}
                    Feature matrix
                    TPOT and all scikit-learn algorithms assume that the features will be numerical
                    and there will be no missing values. As such, when a feature matrix is provided
                    to TPOT, all missing values will automatically be replaced (i.e., imputed) using
                    median value imputation.
                    If you wish to use a different imputation strategy than median imputation, please
                    make sure to apply imputation to your feature set prior to passing it to TPOT.
                target: array-like {n_samples}
                    List of class labels for prediction
                sample_weight: array-like {n_samples}, optional
                    Per-sample weights. Higher weights indicate more importance. If specified,
                    sample_weight will be passed to any pipeline element whose fit() function accepts
                    a sample_weight argument. By default, using sample_weight does not affect tpot's
                    scoring functions, which determine preferences between pipelines.
                groups: array-like, with shape {n_samples, }, optional
                    Group labels for the samples used when performing cross-validation.
                    This parameter should only be used in conjunction with sklearn's Group cross-validation
                    functions, such as sklearn.model_selection.GroupKFold
                Returns
                -------
                self: object
                    Returns a copy of the fitted TPOT object
                """
        self._fit_init()
        features, target = self._check_dataset(features, target, sample_weight)

        self.pretest_X, _, self.pretest_y, _ = train_test_split(features,
                                                                target,
                                                                train_size=min(50, int(0.9 * features.shape[0])),
                                                                test_size=None, random_state=self.random_state)

        # Randomly collect a subsample of training samples for pipeline optimization process.
        if self.subsample < 1.0:
            features, _, target, _ = train_test_split(features, target, train_size=self.subsample, test_size=None,
                                                      random_state=self.random_state)
            # Raise a warning message if the training size is less than 1500 when subsample is not default value
            if features.shape[0] < 1500:
                print(
                    'Warning: Although subsample can accelerate pipeline optimization process, '
                    'too small training sample size may cause unpredictable effect on maximizing '
                    'score in pipeline optimization process. Increasing subsample ratio may get '
                    'a more reasonable outcome from optimization process in TPOT.'
                )

        # Set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state)  # deap uses random
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now()
        self._last_pipeline_write = self._start_datetime
        if not self.meta_fitness_flag:
            self._toolbox.register('evaluate', self._evaluate_individuals, features=features, target=target,
                                   sample_weight=sample_weight, groups=groups)
        else:
            self._toolbox.register('evaluate', self._meta_evaluate_individuals, features=features, target=target,
                                   sample_weight=sample_weight, groups=groups)

        # assign population, self._pop can only be not None if warm_start is enabled
        if not self._pop:
            self._pop = self._toolbox.population(n=self.population_size)

        if self.use_boosting_flag:
            boost_pop = self._boost_population()
            self._pop[:self.boosting_size] = boost_pop

        # Using meta-learning to reduce population
        if self.use_metamodel_flag:
            test_df = create_ranking_df_from_pop(self._pop, self.primitives_to_hash_dic, self.gptree, self._pset,
                                                 self.meta_features_df, self.longest_pipeline_size)
            top_pop_index = rank_pop(self.meta_model, test_df, self.rank_size, self.le)
            top_pop = [self._pop[i] for i in top_pop_index]
            self._pop = top_pop

        def pareto_eq(ind1, ind2):
            """Determine whether two individuals are equal on the Pareto front.
            Parameters
            ----------
            ind1: DEAP individual from the GP population
                First individual to compare
            ind2: DEAP individual from the GP population
                Second individual to compare
            Returns
            ----------
            individuals_equal: bool
                Boolean indicating whether the two individuals are equal on
                the Pareto front
            """
            return np.allclose(ind1.fitness.values, ind2.fitness.values)

        # Generate new pareto front if it doesn't already exist for warm start
        if not self.warm_start or not self._pareto_front:
            self._pareto_front = tools.ParetoFront(similar=pareto_eq)

        # Set lambda_ (offspring size in GP) equal to population_size by default
        if not self.offspring_size:
            self._lambda = self.population_size
        else:
            self._lambda = self.offspring_size

        # Start the progress bar
        if self.max_time_mins:
            total_evals = self.population_size
        else:
            total_evals = self._lambda * self.generations + self.population_size

        self._pbar = tqdm(total=total_evals, unit='pipeline', leave=False,
                          disable=not (self.verbosity >= 2), desc='Optimization Progress')

        try:
            with warnings.catch_warnings():
                self._setup_memory()
                warnings.simplefilter('ignore')
                if self.use_metamodel_flag:
                    mu = self.rank_size
                else:
                    mu = self.population_size
                self._pop, _ = MetaeaMuPlusLambda(
                    population=self._pop,
                    toolbox=self._toolbox,
                    mu=mu,  # mu is rank size in meta learning
                    lambda_=self._lambda,
                    cxpb=self.crossover_rate,
                    mutpb=self.mutation_rate,
                    ngen=self.generations,
                    pbar=self._pbar,
                    halloffame=self._pareto_front,
                    verbose=self.verbosity,
                    meta_model=self.meta_model,
                    per_generation_function=self._test_gen,
                    primitives_to_hash_dic=self.primitives_to_hash_dic,
                    gptree=self.gptree,
                    pset=self._pset,
                    df=self.meta_features_df,
                    le=self.le,
                    use_meta_model_flag=self.use_metamodel_flag,
                    use_meta_selection_flag=self.meta_selection_flag,
                    meta_selection_size=self.meta_selection_size,
                    meta_selection_type=self.meta_selection_type,
                    max_pipeline_size=self.longest_pipeline_size
                )

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write('', file=self._file)
                self._pbar.write('{}\nTPOT closed prematurely. Will use the current best pipeline.'.format(e),
                                 file=self._file)
        finally:
            # clean population for the next call if warm_start=False
            if not self.warm_start:
                self._pop = []
            # keep trying 10 times in case weird things happened like multiple CTRL+C or exceptions
            attempts = 10
            for attempt in range(attempts):
                try:
                    # Close the progress bar
                    # Standard truthiness checks won't work for tqdm
                    if not isinstance(self._pbar, type(None)):
                        self._pbar.close()

                    self._update_top_pipeline()
                    self._summary_of_best_pipeline(features, target)
                    # Delete the temporary cache before exiting
                    self._cleanup_memory()
                    break

                except (KeyboardInterrupt, SystemExit, Exception) as e:
                    # raise the exception if it's our last attempt
                    if attempt == (attempts - 1):
                        raise e
            return self


class MetaTPOTRegressor(MetaTPOTClassifier):
    """
    Main Class - implements MetaTPOTRegressor
    """
    scoring_function = 'neg_mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True

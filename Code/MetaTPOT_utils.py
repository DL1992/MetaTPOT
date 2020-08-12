import os
import pandas as pd
import numpy as np
from deap import creator
from deap.tools import selRandom
from tpot.gp_deap import initialize_stats_dict, varOr, tools


def get_TPOT_pipelines(pipeline_list, json_path, pset):
    pop = []
    for pipeline in pipeline_list:
        # Search the origin full pipeline
        JSONDict = np.load(os.path.join(json_path, f'{pipeline[0]}.npy'), allow_pickle=True).item()
        pipeline_string = JSONDict['Evaluate Pipeline ' + str(int(pipeline[1])) + ':']['TEST PARSING PIPELINE']

        # Create Individual object for Population List
        deap_pipeline = creator.Individual.from_string(pipeline_string, pset)
        pop.append(deap_pipeline)
    return pop


def process_pipeline_hash(hash_pipeline, max_pipeline_size):
    """
    creates a list of primitives representing the pipeline.
    :param hash_pipeline: the hash representation of the pipeline.
    :return: a list of primitives the pipeline is made of. the list is always the size of the max pipeline,
    empty slots are filled with Nan
    """
    primitives = [hash_pipeline[start:start + 40] for start in range(0, len(hash_pipeline), 40)]
    if len(primitives) > max_pipeline_size:
        raise ValueError('too long pipeline')
    while len(primitives) is not max_pipeline_size:
        primitives.append('Nan')
    return primitives


def rank_pop(meta_model, test_df, rank_size, le):
    test_target = test_df[['index', 'instances']] # change back to primitive_1 if breaks
    no_need_att = ['dataset', 'alg_fam', 'fam_score', 'index', 'Features.PearsonCorrelation.NonAggregated',
                   'Labels.PearsonCorrelation.NonAggregated',
                   'FeaturesLabels.PearsonCorrelation.NonAggregated']
    test_dmatrix = prepare_pop_for_rank(test_df, no_need_att, le)
    pred = meta_model.predict(test_dmatrix)
    test_target['pred'] = pred
    return test_target[['index', 'pred']].sort_values(by='pred', ascending=False)['index'][:rank_size]


def prepare_pop_for_rank(test_df, no_nees_att, le):
    test_df = test_df.drop(no_nees_att, axis=1)
    try_data = test_df.select_dtypes(include='object')
    category_columns = try_data.columns
    for feature in category_columns:
        test_df[feature] = le.transform(test_df[feature])
    return test_df


def create_pipeline_row(df, index, pipeline_rep):
    df2 = df.copy()
    primitive_df = pd.DataFrame([pipeline_rep],
                                columns=[f'primitive_{i + 1}' for i in range(len(pipeline_rep))])
    df2 = df2.reset_index(drop=True)
    df2 = pd.concat([df2, primitive_df], axis=1)
    df2['index'] = index
    return df2


def create_ranking_df(meta_learner_df, datasets_metafeatures_df, PIPELINE_SIZE):
    keep_ls = [f'primitive_{i + 1}' for i in range(PIPELINE_SIZE)]
    cols = datasets_metafeatures_df.columns.tolist()
    cols = cols + keep_ls
    meta_learner_df = meta_learner_df[keep_ls]
    meta_learner_df = meta_learner_df.assign(**datasets_metafeatures_df.iloc[0])
    meta_learner_df = meta_learner_df[cols]
    meta_learner_df['index'] = meta_learner_df.index
    return meta_learner_df


def create_ranking_df_from_pop(pop, primitives_to_hash_dic, gptree, pset, df, max_pipeline_size):
    test_df = pd.DataFrame()
    for index, pipeline in enumerate(pop):
        try:
            flatten_pipeline = create_flatten_pipeline('', pipeline, primitives_to_hash_dic, gptree, pset)
        except Exception as e:
            continue
        try:
            pipeline_rep = process_pipeline_hash(flatten_pipeline, max_pipeline_size)
        except ValueError:
            continue
        pipeline_df = create_pipeline_row(df, index, pipeline_rep)
        test_df = test_df.append(pipeline_df)[pipeline_df.columns.tolist()]
    return test_df


def create_flatten_pipeline(flatten_pipeline_str, pipeline, primitives_to_hash_dic, gptree, pset):
    pipeline_as_string = str(pipeline).replace(" ", "")
    for i in range(pipeline.height + 1):
        primitive_name = pipeline[i].name
        if primitive_name != 'CombineDFs':
            flatten_pipeline_str += primitives_to_hash_dic[primitive_name]
            pipeline_as_string = pipeline_as_string[pipeline_as_string.find('(') + 1: (
                    len(pipeline_as_string) - pipeline_as_string[::-1].find(')') - 1)]
        else:
            flatten_pipeline_str += primitives_to_hash_dic[primitive_name]
            sub_pipeline = pipeline_as_string[pipeline_as_string.find('(') + 1: (
                    len(pipeline_as_string) - pipeline_as_string[::-1].find(')') - 1)]  # remove "(" and ")"

            middle = 0
            delimiter_index = [i for i, ltr in enumerate(sub_pipeline) if ltr == ',']
            for deli in delimiter_index:
                if sub_pipeline[:deli].count('(') == sub_pipeline[:deli].count(')'):
                    middle = deli
                    break

            # splitting pipeline for 2 sub pipelines strings
            first_pipeline = sub_pipeline[:middle]
            second_pipeline = sub_pipeline[middle + 1:]

            # creating a pipeline object from the sub pipelines strings
            first_pipeline_ind = gptree.from_string(first_pipeline, pset)
            second_pipeline_ind = gptree.from_string(second_pipeline, pset)

            # longer sub-pipeline first according to pipeline representations
            if first_pipeline_ind.height > second_pipeline_ind.height:
                flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, first_pipeline_ind,
                                                               primitives_to_hash_dic, gptree, pset)
                flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, second_pipeline_ind,
                                                               primitives_to_hash_dic, gptree, pset)
            elif first_pipeline_ind.height < second_pipeline_ind.height:
                flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, second_pipeline_ind,
                                                               primitives_to_hash_dic, gptree, pset)
                flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, first_pipeline_ind,
                                                               primitives_to_hash_dic, gptree, pset)
            else:
                equals = True
                for x, y in zip(first_pipeline_ind, second_pipeline_ind):
                    if x.name not in primitives_to_hash_dic or y.name not in primitives_to_hash_dic:
                        break
                    if primitives_to_hash_dic[x.name] < primitives_to_hash_dic[y.name]:
                        flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, first_pipeline_ind,
                                                                       primitives_to_hash_dic, gptree, pset)
                        flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, second_pipeline_ind,
                                                                       primitives_to_hash_dic, gptree, pset)
                        equals = False
                        break
                    elif primitives_to_hash_dic[x.name] > primitives_to_hash_dic[y.name]:
                        flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, second_pipeline_ind,
                                                                       primitives_to_hash_dic, gptree, pset)
                        flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, first_pipeline_ind,
                                                                       primitives_to_hash_dic, gptree, pset)
                        equals = False
                        break
                if equals:
                    flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, second_pipeline_ind,
                                                                   primitives_to_hash_dic, gptree, pset)
                    flatten_pipeline_str = create_flatten_pipeline(flatten_pipeline_str, first_pipeline_ind,
                                                                   primitives_to_hash_dic, gptree, pset)
                    break
            break
    return flatten_pipeline_str


def _selMetaTournament(individuals, k, tournsize, meta_model, primitives_to_hash_dic,
                       gptree, pset, df, le, max_pipeline_size, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        test_df = create_ranking_df_from_pop(aspirants, primitives_to_hash_dic, gptree, pset, df, max_pipeline_size)
        top_offspring_index = rank_pop(meta_model, test_df, 1, le)
        top_offspring = [aspirants[i] for i in top_offspring_index]
        chosen.append(top_offspring[0])
    return chosen


def MetaeaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar, max_pipeline_size,
                       stats=None, halloffame=None, verbose=0, meta_model=None, per_generation_function=None,
                       primitives_to_hash_dic=None, gptree=None, pset=None, df=None, le=None, use_meta_model_flag=True,
                       use_meta_selection_flag=False, meta_selection_size=10, meta_selection_type='offspring'):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param pbar: processing bar
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param per_generation_function: if supplied, call this function before each generation
                            used by tpot to save best pipeline before each new generation
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    population[:] = toolbox.evaluate(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen=gen - 1, pop=population)

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Using meta-learning to reduce offspring
        if use_meta_model_flag:
            test_df = create_ranking_df_from_pop(offspring, primitives_to_hash_dic, gptree, pset, df, max_pipeline_size)
            top_offspring_index = rank_pop(meta_model, test_df, mu, le)
            top_offspring = [offspring[i] for i in top_offspring_index]
            offspring = top_offspring

        # Using meta-learning as pree-tournament
        if use_meta_selection_flag:
            if meta_selection_type == 'offspring':
                offspring = _selMetaTournament(offspring, mu, meta_selection_size, meta_model=meta_model,
                                               primitives_to_hash_dic=primitives_to_hash_dic, gptree=gptree, pset=pset,
                                               df=df, le=le, max_pipeline_size=max_pipeline_size)
            elif meta_selection_type == 'pop':
                population = _selMetaTournament(population, mu, meta_selection_size, meta_model=meta_model,
                                                primitives_to_hash_dic=primitives_to_hash_dic, gptree=gptree, pset=pset,
                                                df=df, le=le, max_pipeline_size=max_pipeline_size)
            else:
                population = _selMetaTournament(population, mu, meta_selection_size, meta_model=meta_model,
                                                primitives_to_hash_dic=primitives_to_hash_dic, gptree=gptree, pset=pset,
                                                df=df, le=le, max_pipeline_size=max_pipeline_size)
                offspring = _selMetaTournament(offspring, mu, meta_selection_size, meta_model=meta_model,
                                               primitives_to_hash_dic=primitives_to_hash_dic, gptree=gptree, pset=pset,
                                               df=df, le=le, max_pipeline_size=max_pipeline_size)

        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in population:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        offspring = toolbox.evaluate(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = max([halloffame.keys[x].wvalues[1] for x in range(len(halloffame.keys))])
                pbar.write('Generation {0} - Current best internal CV score: {1}'.format(gen, high_score))

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('Generation {} - Current Pareto front scores:'.format(gen))
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('{}\t{}\t{}'.format(
                        int(pipeline_scores.wvalues[0]),
                        pipeline_scores.wvalues[1],
                        pipeline
                    )
                    )
                pbar.write('')

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    if per_generation_function is not None:
        per_generation_function(gen=gen, pop=population)

    return population, logbook

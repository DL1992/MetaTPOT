import glob
import sys
import time
from Code.utils import *
from Code.MetaTPOT_utils import *
from Code.MetaTPOT import MetaTPOTClassifier
from sklearn.preprocessing import LabelEncoder

server_name = sys.argv[1]
set_part = sys.argv[2]
task_type = sys.argv[3]
gen = int(sys.argv[4])
pop = int(sys.argv[5])
knowledge_path = sys.argv[6]
json_path = sys.argv[7]
boosting = int(sys.argv[8])
boosting_percent = float(sys.argv[9])
meta_selection = int(sys.argv[10])
meta_selection_type = sys.argv[11]
meta_selection_size = int(sys.argv[12])
rank_size = int(sys.argv[13])
meta_model_flag = int(sys.argv[14])
dataset_folder_path = sys.argv[15]
results_path = sys.argv[16]
metaFeatures_path = sys.argv[17]
meta_fitness_flag = int(sys.argv[18])


def run_experiments(meta_tpot, X_train, X_test, y_train, y_test, dataset_name, results_path):
    start_time = time.time()
    meta_tpot.fit(X_train, y_train)
    end_time = time.time()
    if not os.path.isdir(os.path.join(results_path, 'pipelines')):
        os.mkdir(os.path.join(results_path, 'pipelines'))
    pipelines_path = os.path.join(results_path, 'pipelines')
    pipelines_dic_file_name = f'{dataset_name}_metatpot_pipelines.npy'
    np.save(os.path.join(pipelines_path, pipelines_dic_file_name), meta_tpot.evaluated_individuals_)
    meta_tpot_test_score = meta_tpot.score(X_test, y_test)
    result_columns = ['gen', 'avg_gen_score', "median_gen_score", 'max_gen_score', 'min_gen_score', 'gen_size']
    meta_tpot_gen_df = pd.DataFrame.from_dict(meta_tpot.gen_results_dic, orient='index', columns=result_columns)
    meta_tpot_gen_df['dataset'] = dataset_name
    meta_tpot_gen_df['runtime'] = end_time - start_time
    meta_tpot_gen_df['test_score'] = meta_tpot_test_score
    meta_tpot_gen_df.to_csv(os.path.join(results_path, f'{dataset_name}_results.csv'), index=False)


def _set_results_folders(results_path, MetaTPOT_args_dic):
    results_path = os.path.join(results_path, 'MetaTPOT_Results')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    if not MetaTPOT_args_dic['boosting'] and not MetaTPOT_args_dic['meta_selection']:
        results_path_to_return = os.path.join(results_path, r'TPOT')
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)
    if MetaTPOT_args_dic['boosting']:
        results_path_to_return = os.path.join(results_path, r'boosting')
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['boosting_percent']}")
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)
    if MetaTPOT_args_dic['meta_selection']:
        results_path_to_return = os.path.join(results_path, r'selection')
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['meta_selection_type']}")
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['meta_selection_size']}")
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)
    if MetaTPOT_args_dic['meta_model_flag']:
        results_path_to_return = os.path.join(results_path, r'ranking')
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['rank_size']}")
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)
    if MetaTPOT_args_dic['meta_fitness_flag']:
        results_path_to_return = os.path.join(results_path, r'fitness')
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)
    if MetaTPOT_args_dic['boosting'] and MetaTPOT_args_dic['meta_selection']:
        results_path_to_return = os.path.join(results_path, r'boosting + selection')
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['meta_selection_type']}")
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['meta_selection_size']}")
        results_path_to_return = os.path.join(results_path_to_return, f"{MetaTPOT_args_dic['boosting_percent']}")
        if not os.path.isdir(results_path_to_return):
            os.makedirs(results_path_to_return)

    return results_path_to_return


def _get_finished_datasets(results_path):
    return set([(file.split('/')[-1]).split('_results')[0] for file in glob.glob(results_path + '/*.csv')])


def run_metaTPOT_experiments(datasets_folder_path, results_path, set_part,
                             datasets_metafeatures_path, task_type='Classification', MetaTPOT_args_dic=None):
    datasets_folder_path = os.path.join(datasets_folder_path, task_type)
    datasets_folder_path = os.path.join(datasets_folder_path, r'split datasets')
    results_path = os.path.join(results_path, task_type)
    PIPELINE_SIZE, _, _, _ = load_stat(path=os.path.join(results_path, 'Processing_results'), task_type=task_type)
    datasets_names = load_datasets(results_path, task_type, set_part)
    ranking_models_folder_path = os.path.join(results_path, r'Ranking_models')
    results_path = _set_results_folders(results_path, MetaTPOT_args_dic)
    datasets_metafeatures_path = os.path.join(datasets_metafeatures_path, task_type)
    datasets_metafeatures_df = load_dataset_metafeatures(datasets_metafeatures_path)
    datasets = set(datasets_metafeatures_df['dataset'].unique())
    cnt = 0
    for dataset in datasets:
        dataset_name = dataset.split('.')[0]
        if dataset_name in datasets_names:
            finshed_datasets = _get_finished_datasets(results_path)
            if dataset_name in finshed_datasets:
                cnt += 1
                continue
            try:
                try:
                    x_train, y_train, x_test, y_test = load_split_dataset(datasets_folder_path, dataset_name)
                except Exception as e:
                    cnt += 1
                    continue
                data_df = datasets_metafeatures_df[datasets_metafeatures_df['dataset'] == dataset]
                ranking_model = load_ranking_model(ranking_models_folder_path, dataset_name, 'pairwise')
                meta_tpot = MetaTPOTClassifier(model=ranking_model, dataset_meta_features_df=data_df,
                                               primitive_to_hash_dic=MetaTPOT_args_dic['Primitive_To_Hash_dict'],
                                               le=MetaTPOT_args_dic['le'],
                                               knowledge_base_path=MetaTPOT_args_dic['knowledge_base_path'],
                                               json_path=MetaTPOT_args_dic['json_path'],
                                               longest_pipeline_size=PIPELINE_SIZE,
                                               rank_size=MetaTPOT_args_dic['rank_size'],
                                               use_meta_model_flag=MetaTPOT_args_dic['meta_model_flag'],
                                               boosting=MetaTPOT_args_dic['boosting'],
                                               boosting_percent=MetaTPOT_args_dic['boosting_percent'],
                                               meta_selection_window_size=MetaTPOT_args_dic['meta_selection_size'],
                                               meta_selection=MetaTPOT_args_dic['meta_selection'],
                                               meta_selection_type=MetaTPOT_args_dic['meta_selection_type'],
                                               n_jobs=MetaTPOT_args_dic['n_jobs'],
                                               generations=MetaTPOT_args_dic['gen'],
                                               population_size=MetaTPOT_args_dic['pop'],
                                               max_time_mins=300,
                                               verbosity=MetaTPOT_args_dic['verbosity'],
                                               meta_fitness_flag=MetaTPOT_args_dic['meta_fitness_flag'])

                meta_tpot.default_config_dict.pop("sklearn.linear_model.SGDClassifier", None)
                run_experiments(meta_tpot=meta_tpot, X_train=x_train, X_test=x_test, y_train=y_train,
                                y_test=y_test, dataset_name=dataset_name, results_path=results_path)
                cnt += 1
            except Exception as err:
                cnt += 1


hash_to_primitive_dic, primitive_to_hash_dic = load_dics(os.path.dirname(os.path.abspath(__file__)),
                                                         task_type='Classification')
le = LabelEncoder()
le.fit(list(hash_to_primitive_dic.keys()))
MetaTPOT_args_dic = {"knowledge_base_path": knowledge_path, "json_path": json_path, "boosting": boosting,
                     "boosting_percent": boosting_percent,
                     "meta_selection": meta_selection, "meta_selection_type": meta_selection_type,
                     "meta_selection_size": meta_selection_size,
                     "Primitive_To_Hash_dict": primitive_to_hash_dic, 'le': le, 'rank_size': rank_size, 'n_jobs': -1,
                     'gen': gen, 'pop': pop, 'verbosity': 0, 'meta_model_flag': meta_model_flag,
                     'meta_fitness_flag': meta_fitness_flag}

run_metaTPOT_experiments(datasets_folder_path=dataset_folder_path,
                         results_path=results_path,
                         datasets_metafeatures_path=metaFeatures_path,
                         MetaTPOT_args_dic=MetaTPOT_args_dic, set_part=set_part)

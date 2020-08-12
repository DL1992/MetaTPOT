import os
import sys
import json
import glob
import pickle
import hashlib
import datetime
import pkgutil
import logging
import operator
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import LabelEncoder as Le
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def load_dics(path, task_type='Classification'):
    """
    loads dics for converting primitives
    :param path: the path where the files are(string ending in '\'
    :param task_type: the task type to load the dic, can be 'Classification' or 'Regression'
    :return: return both dics
    """
    path_to_dic = os.path.join(path, f'{task_type}_hash_To_Primitive_dict.pickle')
    with open(path_to_dic, 'rb') as handle:
        hash_to_primitive_dic = pickle.load(handle)
    path_to_dic = os.path.join(path, f'{task_type}_Primitive_To_Hash_dict.pickle')
    with open(path_to_dic, 'rb') as handle:
        primitive_to_hash_dict = pickle.load(handle)
    return hash_to_primitive_dic, primitive_to_hash_dict


def load_stat(path, task_type='Classification', pre_process_flag=True):
    """
    load all statistic files for use and EDA
    :param path: the path to where the files are
    :param task_type: the task type, either 'Classification' or 'Regression'.
    :param pre_process_flag: boolean if to load the raw statistics or the the metalearner dataframe statistics.
    :return: max_pipeline_length, set of unique pipelines, primitives count dict, pipelines count dic
    """
    pre_process_str = 'pre_process' if pre_process_flag else 'post_process'
    path_to_load = os.path.join(path, f'{task_type}_{pre_process_str}_stat.json')
    with open(path_to_load, 'r') as fp:
        data_dic = json.load(fp)
    path_to_load = os.path.join(path, f'{task_type}_{pre_process_str}_unique_pipelines.pickle')
    with open(path_to_load, 'rb') as fp:
        unique_pipelines_from_pickle = pickle.load(fp)
    path_to_load = os.path.join(path, f'{task_type}_{pre_process_str}_primitives_stat.json')
    with open(path_to_load, 'r') as fp:
        primitives_count_from_json = json.load(fp)
    path_to_load = os.path.join(path, f'{task_type}_{pre_process_str}_pipelines_stat.json')
    with open(path_to_load, 'r') as fp:
        pipeline_counts_from_json = json.load(fp)
    return data_dic['max_pipeline_length'], unique_pipelines_from_pickle, \
           primitives_count_from_json, pipeline_counts_from_json


def load_datasets(result_path, task_type, set_part):
    with open(os.path.join(result_path, f'{task_type}_datasets_set_{set_part}.pickle'), 'rb') as p_file:
        datasets = pickle.load(p_file)
    return datasets


def load_dataset_metafeatures(datasets_metafeatures_path):
    for filename in os.listdir(datasets_metafeatures_path):
        df = pd.read_csv(os.path.join(datasets_metafeatures_path, filename))
    return df


def load_ranking_model(ranking_models_folder_path, dataset_name, ranking_function='pairwise'):
    """
    Loading a trained meta-learner model
    :param ranking_models_folder_path: path to the ranking models folder
    :param dataset_name: the dataset the metalearner was trained for
    :param ranking_function: which type of ranking function was used to train the model
    :return: the ranking metamodel, raise ERROR if couldnt load model
    """
    if ranking_function == 'pairwise':
        ranking_models_folder_path = os.path.join(ranking_models_folder_path, 'Pairwise')
    elif ranking_function == 'map':
        ranking_models_folder_path = os.path.join(ranking_models_folder_path, 'Map')
    else:
        ranking_models_folder_path = os.path.join(ranking_models_folder_path, 'ndcg')
    directory = os.fsencode(ranking_models_folder_path)
    for filename in os.listdir(directory):
        if dataset_name in str(filename):
            model = jb.load(os.path.join(directory, filename))
            return model
    raise RuntimeError(f'couldnt load ranking model for {dataset_name}')


def pipeline_to_json(primitive_indexing, hash_two_arguments, pipeline):
    """
    Converting TPOT pipeline to JSON as a dictionary.

    :param primitiveIndexing:
    :param hashTwoArguments: List. list of all primitive that have two inputs.
    :param pipeline: String. TPOT representation of pipeline
    :return: Dictionary. JSON form Dictionary containing all the pipeline data.
    """
    pipelineDic = {}

    pipeline = pipeline.replace(" ", "")

    step = pipeline[:pipeline.find(')')].count('(')

    while len(pipeline) != 0:

        # extract primitive name
        primitiveEnd = pipeline.find("(")

        if primitiveEnd != -1:
            primitiveName = pipeline[:primitiveEnd]
        else:
            pipeline = pipeline.replace(")", "").replace(",", "")

            #  input matrix
            pipelineDic['STEP {0}'.format(step)] = {
                "1.INPUT": pipeline}

            break

        if primitive_indexing[primitiveName] not in hash_two_arguments:

            # extract primitive hyperparam
            reverse = pipeline[::-1][1:]

            #  check if this is the last primitive
            if reverse.find(")") != -1:
                hyperparameterStart = len(reverse) - reverse.find(")")
                hyperParam = pipeline[hyperparameterStart + 1:len(pipeline) - 1]  # +1 for ','
                hyperParam = hyperParam.split(',')  # [1:]
            else:
                if pipeline.find(',') != -1:
                    hyperparameterStart = pipeline.find(',')
                    hyperParam = pipeline[hyperparameterStart:len(pipeline) - 1]
                    hyperParam = hyperParam.split(',')
                else:
                    hyperParam = []
                    hyperparameterStart = len(pipeline) - 1

            hyperParam = [x[x.find('__') + 2:] for x in hyperParam]

            #  Writing to main Dictionary
            pipelineDic['STEP {0}'.format(step)] = {
                "1.PRIMITIVE": primitiveName, "2.HYPERPARAMETERS": hyperParam}

            #  Step
            step -= 1
            pipeline = pipeline[primitiveEnd + 1:hyperparameterStart]

        else:

            input = pipeline[len(primitiveName) + 1:-1]

            delimiterIndex = [i for i, ltr in enumerate(input) if ltr == ',']

            findDelimiter = 0

            #  find middle of the subPipeline
            for delimiter in delimiterIndex:
                if input[:delimiter].count('(') == input[:delimiter].count(')'):
                    findDelimiter = delimiter
                    break

            firstSubPipeline = input[:findDelimiter]
            secondSubPipeline = input[findDelimiter + 1:]

            inputOneJsonPipelineDic = pipeline_to_json(primitive_indexing, hash_two_arguments, firstSubPipeline)
            inputTwoJsonPipelineDic = pipeline_to_json(primitive_indexing, hash_two_arguments, secondSubPipeline)

            #  Writing to json Dictionary
            pipelineDic["STEP {0}".format(step)] = {
                "1.PRIMITIVE": primitiveName, "2.FIRST INPUT": inputOneJsonPipelineDic,
                "3.SECOND INPUT": inputTwoJsonPipelineDic}

            break

    return pipelineDic


def insert_pipeline_to_dic(primitive_indexing, hash_two_arguments, dict, pipeline_origin, pipeline_score,
                           pipeline_index):
    """
    Insert TPOT pipeline as a string and score to the given dictionary at the given index.

    :param primitive_indexing: Dictionary. Dictionary containing all the one argument primitive,
                                            and for each primitive his hash as a value.
    :param hash_two_arguments: Dictionary. Dictionary containing all the one argument primitive,
                                              and for each primitive his hash as a value.
    :param dict: Dictionary. Dictionary that the pipeline will be save to.
    :param pipeline_origin: String. TPOT representation of pipeline.
    :param pipeline_score: Int. Pipeline score value.
    :param pipeline_index: Int. Index of the given pipeline at the dictionary.
    :return:
    """
    dict["Evaluate Pipeline {0}:".format(pipeline_index)] = pipeline_to_json(primitive_indexing, hash_two_arguments,
                                                                             pipeline_origin)

    #  Write pipelineScore, pipelineIndex
    dict["Evaluate Pipeline {0}:".format(pipeline_index)]["TEST SCORE"] = pipeline_score

    dict["Evaluate Pipeline {0}:".format(pipeline_index)][
        "TEST PARSING PIPELINE"] = pipeline_origin


def breakDictionaryToArray(primitive_indexing, hash_two_arguments, pipeline):
    """
    Get pipeline as a Dict and return array of string each representing
    step(primitive).

    :param primitive_indexing: Dictionary. Dictionary containing all the one argument primitive,
                                            and for each primitive his hash as a value.
    :param hash_two_arguments: Dictionary. Dictionary containing all the one argument primitive,
                                              and for each primitive his hash as a value.
    :param pipeline: Dictionary. JSON representation of pipeline.
    :return: List. List of string, each item represent primitive according to the steps.
    """
    originalPipeline = pipeline
    # Trying to Parse the pipeline
    try:

        primitiveArr = []
        size = 0

        if type(pipeline) is dict:
            for step, primitive in pipeline.items():
                if isinstance(primitive, float):
                    continue
                elif '1.INPUT' in primitive:
                    if primitive['1.INPUT'] not in primitive_indexing:
                        raise KeyError(
                            "breakDictionaryToArray couldn't classified one of the primitive to the primitive "
                            "list.\nThe primitive is'{}'\nAt Pipeline:\n{}".format(primitive['1.INPUT'],
                                                                                   json.dumps(originalPipeline,
                                                                                              sort_keys=True,
                                                                                              indent=4)))
                    primitiveArr.append(primitive['1.INPUT'])
                    size += 1
                elif '1.PRIMITIVE' in primitive:

                    if primitive['1.PRIMITIVE'] not in primitive_indexing:
                        raise KeyError(
                            "breakDictionaryToArray couldn't classified one of the primitive to the primitive "
                            "list.\nThe primitive is'{}'\nAt Pipeline:\n{}".primitive['1.PRIMITIVE'],
                            json.dumps(originalPipeline, sort_keys=True, indent=4))

                    primitiveName = primitive['1.PRIMITIVE']

                    if hashlib.sha1(primitiveName.encode('utf-8')).hexdigest() not in hash_two_arguments:  # changed

                        primitiveArr.append(primitiveName)
                        size += 1

                    else:

                        firstInput = primitive['2.FIRST INPUT']
                        secondInput = primitive['3.SECOND INPUT']

                        primitiveArr.append(primitiveName)
                        size += 1

                        firstPrimitiveArr, firstSize = breakDictionaryToArray(primitive_indexing, hash_two_arguments,
                                                                              firstInput)
                        secondPrimitiveArr, secondSize = breakDictionaryToArray(primitive_indexing, hash_two_arguments,
                                                                                secondInput)

                        for Fprimitive in firstPrimitiveArr:
                            if Fprimitive not in primitive_indexing:
                                raise KeyError(
                                    "The primitive {} is not at the given dictianary.".format(Fprimitive))

                        for Sprimitive in secondPrimitiveArr:
                            if Sprimitive not in primitive_indexing:
                                raise KeyError(
                                    "The primitive {} is not at the given dictianary.".format(Sprimitive))

                        #  check which of the sub-pipeline are longer
                        if firstSize > secondSize:
                            primitiveArr = primitiveArr + firstPrimitiveArr + secondPrimitiveArr
                        elif firstSize < secondSize:
                            primitiveArr = primitiveArr + secondPrimitiveArr + firstPrimitiveArr
                        else:
                            equals = True

                            #  iterate each element of the equal sub-pipelines
                            for x, y in zip(firstPrimitiveArr, secondPrimitiveArr):

                                #  found that firstPrimitiveArr
                                if str(primitive_indexing[x]) < str(primitive_indexing[y]):
                                    primitiveArr = primitiveArr + firstPrimitiveArr + secondPrimitiveArr
                                    equals = False
                                    break
                                elif primitive_indexing[x] > primitive_indexing[y]:
                                    primitiveArr = primitiveArr + secondPrimitiveArr + firstPrimitiveArr
                                    equals = False
                                    break

                            if equals:
                                primitiveArr = primitiveArr + firstPrimitiveArr + secondPrimitiveArr

                        size = size + firstSize + secondSize
        else:
            if pipeline not in primitive_indexing:
                raise KeyError("breakDictionaryToArray couldn't classified the pipeline to ['float', 'PRIMITIVE', "
                               "'INPUT'].\nThe pipeline is'{}', and it isn't at the primitive list!".format(pipeline))
            primitiveArr.append(pipeline)
            size += 1

        return primitiveArr, size

    except KeyError as e:
        print(str(e))
    except Exception as e:
        print(str(e))
        print("breakDictionaryToArray Fail to parse the pipeline\n\n{}\n".format(
            json.dumps(pipeline, sort_keys=True, indent=4)))


def add_flatten_pipelines_to_dataFrame(primitive_indexing, hash_two_arguments, data_frame_result, pipeline_dict, score,
                                       pipeline_number, dataset_name):
    """
    Transform the given pipeline to hash and add to dataframe.

    :param primitive_indexing: Dictionary. Dictionary containing all the one argument primitive,
                                            and for each primitive his hash as a value.
    :param hash_two_arguments: Dictionary. Dictionary containing all the one argument primitive,
                                              and for each primitive his hash as a value.
    :param data_frame_result: Dataframe. Dataframe to save the pipeline data(flatten pipeline).
    :param pipeline_dict: Dictionary. Dictionary representing the pipeline.
    :param score: Int. Pipeline score value.
    :param pipeline_number: Int. Pipeline number according to dictionary.
    :param dataset_name: String. Dataset name.
    :return:
    """
    try:
        # Transform given pipeline to primitive array
        primitiveArr, _ = breakDictionaryToArray(primitive_indexing, hash_two_arguments,
                                                 pipeline_dict['Evaluate Pipeline {}:'.format(pipeline_number)])

        if len(primitiveArr) == 0:
            raise ValueError(
                ("Primitive array failed to create (size 0).\n for data set{} and pipeline No.UNKNOWN.".format(
                    dataset_name)))

        pipelineIndex = pipeline_number
        pipelineScore = score
        flatten_pipeline = ""

        try:
            for primitive in primitiveArr:
                flatten_pipeline += primitive_indexing[primitive]

            dataFrameResult = data_frame_result.append({'dataset_Name': dataset_name, 'flattenPipeline':
                flatten_pipeline, 'pipelineNumber': pipelineIndex, 'score': pipelineScore}, ignore_index=True)
            return dataFrameResult

        except Exception as e:
            print("\nFail to add flatten pipeline to dataframe.\nflattenpipeline: {}\n".format(flatten_pipeline))
            print(str(e))
    except KeyError as e:
        print("\nOne or more primitive in the pipeline array was not exist at the dictionary.\ncant create "
              "flatten pipeline.\nprimitive:{}\nprimitiv array:{} ".format(primitive, *primitiveArr, sep=", "))
    except Exception as e:
        print(str(e))


def delete_text_from_files(path_to_dir, string_list, delete_empty_lines=False):
    """
    Delete each line if it equal to one of the string at the given
    string_list for each file at the given path.

    :param path_to_dir: Path(String), absolute path to files directory.
    :param string_list: List. containing all String that need to be deleted.
    :param delete_empty_lines: Boolean. if True, delete empty lines.
    :return:
    """
    for importer, package_name, _ in pkgutil.iter_modules([path_to_dir]):
        full_package_name = '%s\\%s' % (path_to_dir, package_name)
        if full_package_name not in sys.modules:

            with open(f"{full_package_name}.py", "r") as f:
                lines = f.readlines()
            with open(f"{full_package_name}.py", "w") as f:
                for line in lines:
                    if delete_empty_lines and line == "\n":
                        pass
                    elif line.strip('\n') not in string_list or line == "\n":
                        f.write(line)


def add_text_to_file(path_to_file, string, start=False):
    """
    Add each line to the end of the file at the given path.

    :param path_to_file: Path(String), absolute path to file.
    :param string: String. string to add to file at the end.
    :param start: Boolean. indicate if the string suppose to add at the start of the file.
    :return:
    """
    line_to_add = string.split('\n')

    if not start:
        if path_to_file not in sys.modules:
            with open(path_to_file, "a") as f:
                for line in line_to_add:
                    f.write(line)
                    f.write('\n')
            f.close()

    else:
        with open(path_to_file, "r") as f:
            lines = f.readlines()
        with open(path_to_file, "w") as f:
            for line in line_to_add:
                f.write(line)
                f.write('\n')
            for line in lines:
                f.write(line)
                f.write('\n')
        f.close()


def add_text_to_files(path_to_dir, string, start=False):
    """
    Add each line to the end of each file at the given path.

    :param path_to_dir: Path(String), absolute path to files directory.
    :param string: String. string to add to each file at the end.
    :param start: Boolean. indicate if the string suppose to add at the start of the file.
    :return:
    """
    line_to_add = string.split('\n')
    files_list = glob.glob(path_to_dir)

    if not start:
        for file in files_list:
            full_package_name = os.path.join(os.getcwd(), file)

            with open(full_package_name, "a") as f:
                for line in line_to_add:
                    f.write(line)
                    f.write('\n')
            f.close()

    else:
        for file in files_list:
            full_package_name = os.path.join(os.getcwd(), file)
            with open(full_package_name, "r") as f:
                lines = f.readlines()
            with open(full_package_name, "w") as f:
                for line in line_to_add:
                    f.write(line)
                    f.write('\n')
                for line in lines:
                    f.write(line)
            f.close()


def run_python_codes(path_to_dir):
    """
    Run each of the python code in the given directory.

    :param path_to_dir: Path(String), absolute path to files directory.
    :return:
    """
    # myFiles = os.listdir(path_to_dir)
    # for f in myFiles:
    #     module = importer.find_module(f).load_module(f)
    #     print(module)
    for index, (importer, package_name, _) in enumerate(pkgutil.iter_modules([path_to_dir])):
        # full_package_name = '%s\\%s' % (path_to_dir, package_name)
        module = importer.find_module(package_name).load_module(package_name)
        print(module)


def split_and_save_dataset(dataset_path, destination_path, train_test_split_size=0.8):
    """
    Split dataset according to given split ratio.
    the split dataset will be save at the destination path.
    :param datast_path: String. Loaction of the dataset.
    :param destination_path:  String. split dataset destination.
    :param train_test_split_size: Double. Split ratio.
    :return:
    """

    if os.path.isfile(dataset_path) == 0:
        raise ValueError("File not Exist")
    if os.path.isdir(destination_path) == 0:
        os.makedirs(destination_path)

    data_set_type = dataset_path[-3:]
    data_set_name = os.path.basename(dataset_path).split('.')[0]

    if data_set_type == 'txt' or data_set_type == 'csv':
        data_set = pd.read_csv(dataset_path, sep=",")
    elif data_set_type == 'dat':
        data_set = pd.read_csv(dataset_path, sep='\t')
    else:
        raise ValueError("Dataset extension not valid")

    X, y = data_set.iloc[:, :-1], data_set.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_test_split_size)

    path_to_save = os.path.join(destination_path, data_set_name)

    np.save(f'{path_to_save}_X_test.npy', X_test)
    np.save(f'{path_to_save}_y_test.npy', y_test)
    np.save(f'{path_to_save}_X_train.npy', X_train)
    np.save(f'{path_to_save}_y_train.npy', y_train)


def split_and_save_datasets(datasets_path, destination_path, train_test_split_size=0.8):
    """
    Split datasets according to given split ratio.
    the split dataset will be save at the destination path.
    :param datasts_path: String. Loaction of the dataset.
    :param destination_path:  String. split dataset destination.
    :param train_test_split_size: Double. Split ratio.
    :return:
    """

    if not os.path.isdir(datasets_path):
        raise ValueError("path that given isn't a directory")

    if not os.path.isdir(destination_path):
        raise ValueError("path that given isn't a directory")

    for filename in glob.glob(os.path.join(datasets_path, '*.*')):
        split_and_save_dataset(os.path.join(datasets_path, filename), destination_path, train_test_split_size)


def convert_item_to_function(path_to_file, return_item, function_name):
    inside_function = False

    with open(path_to_file, "r") as f:
        lines = f.readlines()

    with open(path_to_file, "w") as f:
        for line in lines:
            if return_item in line:
                inside_function = True
                f.write(f"def {function_name}():\n")
                f.write(f"\t{line}")

            elif inside_function and line == '\n':
                f.write(f"\treturn {return_item}\n")
                inside_function = False

            elif inside_function:
                f.write(f'\t{line}')

            else:
                f.write(f"{line}")


def load_split_dataset(dataset_path, dataset_name):
    """
    Load existing split train and test dataset.

    :param dataset_path: String. Path to split dataset loacation.
    :param dataset_name: String. Dataset name.
    :return: training_features, training_target, testing_features, testing_target.
    """
    training_features = np.load(os.path.join(dataset_path, f'{dataset_name}_X_train.npy'))
    testing_features = np.load(os.path.join(dataset_path, f'{dataset_name}_X_test.npy'))

    training_target = np.load(os.path.join(dataset_path, f'{dataset_name}_y_train.npy'))
    testing_target = np.load(os.path.join(dataset_path, f'{dataset_name}_y_test.npy'))

    return training_features, training_target, testing_features, testing_target


def evaluate_pipeline(dataset_path, dataset_name, pipeline, Task='classification'):
    """
    Evaluate splited dataset on sklearn pipeline.
    :param dataset_path: String. Location of split dataset.
    :param pipeline: Pipeline. SK-Learn pipelien Object.
    :return: train_accuracy, test_accuracy.
    """

    training_features, training_target, testing_features, testing_target = load_split_dataset(dataset_path,
                                                                                              dataset_name)

    if Task == 'classification':
        try:
            pipeline.fit(training_features, training_target)

        except ValueError as e:
            training_features = np.nan_to_num(training_features)
            training_target = np.nan_to_num(training_target)
            testing_features = np.nan_to_num(testing_features)
            testing_target = np.nan_to_num(testing_target)
            pipeline.fit(training_features, training_target)

        predictions = pipeline.predict(testing_features)

        train_accuracy = accuracy_score(training_target, pipeline.predict(training_features))
        test_accuracy = accuracy_score(testing_target, predictions)

        return train_accuracy, test_accuracy

    elif Task == 'regression':
        try:
            pipeline.fit(training_features, training_target)

        except ValueError:
            from sklearn.preprocessing import Imputer

            imputer = Imputer(strategy="median")
            imputer.fit(training_features)
            training_features = imputer.transform(training_features)
            testing_features = imputer.transform(testing_features)
            try:
                pipeline.fit(training_features, training_target)
            except ValueError:
                print(dataset_name)
                return None, None
        predictions = pipeline.predict(testing_features)
        train_mse = mean_squared_error(training_target, pipeline.predict(training_features))
        test_mse = mean_squared_error(testing_target, predictions)

        return train_mse, test_mse

    else:
        raise ValueError("Task must be Clasification or Regression.")


def save_to_dataframe(dataset_path, dict):
    """

    :param dataset_path: Sting. path to Dataset destination.
    :param dict: Dictionary. Dictionary contain the data that suppose to inset the dataframe.
    :return:
    """
    if os.path.isfile(dataset_path):
        try:
            df = pd.read_csv(dataset_path, index_col=0)
        except ValueError:
            print(f"{dataset_path} Failed to load DataFrame")
    else:
        column_list = dict.keys()
        df = pd.DataFrame(columns=column_list)

    df = df.append(dict, ignore_index=True)

    df.to_csv(dataset_path)


def get_pipelines_statistic(dataframe, column_name, dest_path, primitive_to_hash_dict, hash_two_arguments,
                            list_of_primitive_to_remove, task):
    if task not in ['Classification', 'Regression']:
        raise ValueError("task that given is not valid.")

    if not os.path.isdir(dest_path):
        raise ValueError("the path that given is not valid.")

    if column_name not in dataframe.columns:
        raise ValueError("the given column name not in dataframe")

    dest_path = dest_path+'\\'
    result_best_pipeline = dataframe[column_name].tolist()

    total_number_of_primitive = 0
    minimum_pipeline_length = float('inf')
    maximum_pipeline_length = 0

    similar_pipeline_dict = {}
    temp_dict = {}

    primitive_count = {}
    for primitive_name in primitive_to_hash_dict.keys():
        primitive_count[primitive_name] = 0

    for pipeline in result_best_pipeline:

        pipeline_as_json = pipeline_to_json(primitive_to_hash_dict, hash_two_arguments, pipeline)
        primitive_list = breakDictionaryToArray(primitive_to_hash_dict, hash_two_arguments, pipeline_as_json)[0]

        flatten_pipeline = ""

        for primitive in primitive_list:
            flatten_pipeline += primitive_to_hash_dict[primitive]

        if flatten_pipeline in temp_dict:
            similar_pipeline_dict[temp_dict[flatten_pipeline]] = similar_pipeline_dict[temp_dict[flatten_pipeline]] + 1

        else:
            temp_dict[flatten_pipeline] = pipeline
            similar_pipeline_dict[pipeline] = 1

        clean_primitive_list = [value for value in primitive_list if value not in list_of_primitive_to_remove]

        if len(clean_primitive_list) > maximum_pipeline_length:
            maximum_pipeline_length = len(clean_primitive_list)
        if len(clean_primitive_list) < minimum_pipeline_length:
            minimum_pipeline_length = len(clean_primitive_list)

        total_number_of_primitive = total_number_of_primitive + len(clean_primitive_list)

        for primitive_name in clean_primitive_list:
            primitive_count[primitive_name] = primitive_count[primitive_name] + 1

    normalize_primitive_dict = {}
    for primitive_name, primitive_number in primitive_count.items():
        normalize_primitive_dict[primitive_name] = primitive_number / total_number_of_primitive

    sort_normalize_primitive_dict = sorted(normalize_primitive_dict.items(), key=operator.itemgetter(1), reverse=True)

    metadata = {'average_pipeline_length': (total_number_of_primitive / len(result_best_pipeline)),
                'minimum_pipeline_length': minimum_pipeline_length,
                'maximum_pipeline_length': maximum_pipeline_length
                }
    sort_similar_pipeline_dict = sorted(similar_pipeline_dict.items(), key=lambda x: x[1], reverse=True)

    with open(f'{dest_path+task+column_name}_Statistic_of_primitive.pickle', 'wb') as handle:
        pickle.dump(sort_normalize_primitive_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dest_path+task+column_name}_Statistic_of_pipelines(ave,min,max).pickle', 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{dest_path+task+column_name}_Statistic_of_pipelines.pickle', 'wb') as handle:
        pickle.dump(sort_similar_pipeline_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print()


def datasets_pre_processing(origin_dir_path, dest_dir_path, task):
    if not os.path.isdir(origin_dir_path) or not os.path.isdir(dest_dir_path):
        raise ValueError("path that given not valid.")

    if task != 'regression' and task != 'classification':
        raise ValueError("task that given not valid.")

    # create log file
    logger = logging.getLogger(f'utils.datasets_pre_processing_{task}')
    logger.setLevel(logging.DEBUG)
    logger_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Loggers')
    logger_file = os.path.join(logger_file, f'utils.datasets_pre_processing_{task}'
    f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')
    f_handler = logging.FileHandler(logger_file)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.info("datasets_pre_processing.")

    for index, filename in enumerate(glob.glob(os.path.join(origin_dir_path, '*.*'))):
        try:
            full_file_name = os.path.split(filename)[1]

            splited_name = full_file_name.split(".")
            dataset_name = splited_name[0]
            dataset_type = splited_name[1]

            print(f"{dataset_name}")

            if dataset_type == 'txt' or dataset_type == 'csv':
                dataset = pd.read_csv(filename, sep=",")

            elif dataset_type == 'dat':
                dataset = pd.read_csv(filename, sep='\t')
                dataset = dataset.drop(dataset.columns[0], axis=1)

            #  ------------- change '?' to NaN (all dataset) -------------
            dataset.replace('?', np.nan).to_csv("temp.csv", index=False)
            dataset = pd.read_csv('temp.csv')
            print(f"{dataset_name}: done - changing '?' to NaN.")

            #  -------------- split to train and test -------------
            X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
            print(f"{dataset_name}: sone - split to X and y.")

            #  ------------- change Dummies only at attributes(X) -------------
            str_columns_x = X.select_dtypes(include=['object']).columns  # get all "X" object(string) columns
            X = pd.get_dummies(X, prefix=str_columns_x)
            print(f"{dataset_name}: done - change Dummies only at attributes(X).")

            if y.dtype == 'object':
                le = Le()
                le.fit(list(set(y)))
                y = le.transform(y)
                print(f"{dataset_name}: done - label-encoder object only at target(y).")

            print(f"{dataset_name}: done - pre processing.")

            X['target'] = y

            print(f"{dataset_name}: done - combine X and y")

            try:
                X.to_csv(os.path.join(dest_dir_path, f"{dataset_name}.csv"), index=False)
                logger.info(f"{dataset_name} pre-processing done successfully and saved.")
                print(f"{dataset_name}: saved.")

            except ValueError as e:
                logger.error(f"{dataset_name} pre-processing done successfully and failed to save dataset.")
                print(f"{dataset_name}: -------------------- save failed. ({e})")

            print()

        except ValueError as e:
            logger.error(f"{dataset_name} pre-processing fail.")
            print(f"{dataset_name}: pre-processing and save failed. ({e})")

    os.remove("temp.csv.")
    logger.error(f"remove temp file.")
    print(f"remove temp file.")

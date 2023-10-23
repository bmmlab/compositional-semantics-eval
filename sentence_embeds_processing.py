## A module with some basic sentence processing functions
## James Fodor 2023

import numpy as np
import re

# Function to load sentences or pairs of sentences from text file
def load_sentences(file_path: str, pairs: bool):
    """ Loads a set of sentences into a dictionary.

    Args:
        file_path (str): path to sentences file
        pairs (bool): load paired (True) or unpaired (False) sentences

    Returns:
        dict: dictionary of sentence pairs
    """
    
    # Open dataset file
    data_dict = {}
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    # Load data into dictionary
    if pairs==True: # for pairs of sentences
        # Save all sentence pairs into dictionary
        data_dict = {}
        i=1
        for item in lines: 
            item = item.strip()
            item_list = item.split(';')
            similarity = float(item_list[2])
            sent_pair = (item_list[0],item_list[1])
            data_dict[i] = [item_list[0],item_list[1],similarity]
            i=i+1
    elif pairs==False: # for single sentences
        data_dict = {}
        i=1
        for item in lines: 
            item = item.strip()
            data_dict[i] = item
            i=i+1
    
    return data_dict


# Function to load a specific word embedding model
def import_word_model(model_path: str):
    """ Imports an embedding model, storing it in a dictionary.

    Args:
        model_path (str): path to embeddings

    Returns:
        dict: word embeddings dictionary
    """
    
    # open relevant file
    with open(model_path, encoding='utf-8') as file:
        lines = [line.rstrip('\n') for line in file]
    
    # create word dictionary for specific model
    model_dict = {}  
    for line in lines:
        word_list = line.split()
        word = word_list[0]
        embedding_list = [float(x) for x in word_list[1:-1]] # store embeddings
        embedding_np = np.array(embedding_list)
        model_dict[word] = embedding_np
        
    return model_dict

# Convert sentence string to list of tokens
def tokenise_sentence(sentence: str, stop_words: list):
    """ Returns a list of tokens in a sentence.

    Args:
        sentence (str): sentence to tokenise
        stop_words (list): stop words to exclude

    Returns:
        list: list of tokens
    """
    
    token_list = []
    sentence_list = sentence.lower().split()
    for word in sentence_list:
        if word not in stop_words:
            word = re.sub(r'[^\w\s]', '', word) # remove punctuation
            token_list.append(word)
    return token_list


# Calculate cosine similarity between two embeddings
def cosine_sim(embed_1: list, embed_2: list):
    """ Returns the cosine similarity (-1 to 1) between two embeddings, inputted as vectors.

    Args:
        embed_1 (list): word embedding 1
        embed_2 (list): word embedding 2

    Returns:
        float: cosine similarity
    """
    if np.dot(embed_1,embed_2) == 0:
        similarity = 0 # don't normalise if similarity is zero
    else:
        similarity = np.dot(embed_1,embed_2)/(np.linalg.norm(embed_1)*np.linalg.norm(embed_2))
        #similarity, _ = spearmanr(embed_1, embed_2)
    return similarity


# Deal with tokens that can't be found in ConceptNet but are present in some of the sentence datasets
def replace_tricky_tokens(token_list: list):
    """ Replace tokens which are not found in ConceptNet, but are present in sentence sets.

    Args:
        token_list (list): list of tokens, some of which will be replaced

    Returns:
        list: updated list of tokens after replacement
    """
    token_list = [token.replace('tshirt', 't_shirt') for token in token_list]
    token_list = [token.replace('oclock', 'o\'clock') for token in token_list]
    token_list = [token.replace('lorises', 'loris') for token in token_list]
    token_list = [token.replace('rightwing', 'right_wing') for token in token_list]
    token_list = [token.replace('businessmens', 'businessmen') for token in token_list]
    token_list = [token.replace('companys', 'company') for token in token_list]
    token_list = [token.replace('absented', 'absent') for token in token_list]
    token_list = [token.replace('gokart', 'go_kart') for token in token_list]
    token_list = [token.replace('volleyballs', 'volleyball') for token in token_list]
    token_list = [token.replace('bellbottoms', 'bell_bottom') for token in token_list]
    token_list = [token.replace('tball', 't_ball') for token in token_list]
    token_list = [token.replace('backbends', 'backbend') for token in token_list]
    token_list = [token.replace('daschunds', 'daschund') for token in token_list]
    token_list = [token.replace('snowsuits', 'snowsuit') for token in token_list]
    token_list = [token.replace('ponys', 'ponies') for token in token_list]
    token_list = [token.replace('basket_ball', 'basketball') for token in token_list]
    token_list = [token.replace('foot_baller', 'footballer') for token in token_list]
    token_list = [token.replace('openair', 'open_air') for token in token_list]
    token_list = [token.replace('gryffindors', 'gryffindor') for token in token_list]
    token_list = [token.replace('nevilles', 'neville') for token in token_list]
    token_list = [token.replace('hagrids', 'hagrid') for token in token_list]
    token_list = [token.replace('remembrall', 'reminder') for token in token_list]
    token_list = [token.replace('malfoys', 'malfoy') for token in token_list]
    token_list = [token.replace('caughty', 'caught') for token in token_list]
    token_list = [token.replace('slytherins', 'slytherin') for token in token_list]
    token_list = [token.replace('whod', 'who') for token in token_list]
    token_list = [token.replace('hary', 'harry') for token in token_list]
    token_list = [token.replace('hermiones', 'hermione') for token in token_list]
    token_list = [token.replace('enrolls', 'enroll') for token in token_list]
    token_list = [token.replace('mispronounces', 'mispronounce') for token in token_list]
    token_list = [token.replace('allocates', 'allocate') for token in token_list]
    token_list = [token.replace('automates', 'automate') for token in token_list]
    token_list = [token.replace('absents', 'absent') for token in token_list]
    token_list = [token.replace('pleds', 'pled') for token in token_list]
    token_list = [token.replace('subtracts', 'subtract') for token in token_list]
    token_list = [token.replace('poorlyequipped', 'poorly_equipped') for token in token_list]
    token_list = [token.replace('selfcontained', 'self_contained') for token in token_list]
    token_list = [token.replace('bunchs', 'bunch') for token in token_list]
    token_list = [token.replace('stirfried', 'stir_fried') for token in token_list]
    token_list = [token.replace('mildtasting', 'mild_tasting') for token in token_list]
    token_list = [token.replace('vehiclerelated', 'vehicle_related') for token in token_list]
    token_list = [token.replace('submerges', 'submerge') for token in token_list]
    token_list = [token.replace('selfdefense', 'self_defense') for token in token_list]
    token_list = [token.replace('brightlycolored', 'brightly_colored') for token in token_list]
    token_list = [token.replace('illomen', 'ill_omen') for token in token_list]
    token_list = [token.replace('twobladed', 'two_bladed') for token in token_list]
    token_list = [token.replace('blowholes', 'blowhole') for token in token_list]
    token_list = [token.replace('brightlycolored', 'brightly_colored') for token in token_list]
    token_list = [token.replace('cameramans', 'cameraman') for token in token_list]
    return token_list


# Converts a dictionary of word tokens to a numpy matrix of word embeddings
def get_token_embeds(embeddings_matrix: np.ndarray, lemmatizer: object, token_list: list):
    """ Converts a list of tokens to a matrix of corresponding word embeddings.

    Args:
        embeddings_matrix (np.ndarray): matrix of pre-loaded word embeddings
        lemmatizer (object): word lemmatizer
        token_list (list): list of tokens to get embeddings for

    Returns:
        np.ndarray: matrix of word embeddings for all tokens in list
    """
    
    embed_dim = len(embeddings_matrix['man'])
    token_embeds_matrix = np.empty((0,embed_dim), float) # create storage array
    token_list = replace_tricky_tokens(token_list) # replace tokens not found in conceptnet with equivalents

    for token in token_list: # stack all token embeddings into matrix
        try:
            word_embedding = embeddings_matrix[token] # get word embed
            token_embeds_matrix = np.vstack([token_embeds_matrix, word_embedding])
        except KeyError: # try lemmatised version of word
            lemmatised_token = lemmatizer.lemmatize(token,'v')
            try:
                word_embedding = embeddings_matrix[lemmatised_token]
                token_embeds_matrix = np.vstack([token_embeds_matrix, word_embedding])
            except KeyError:
                print('cant find ',token) # if all else fails
    return token_embeds_matrix


# Loads a set of sentence similarities, with the right code depending on whether it is a paired or non-paired list
def load_set_of_sentences(dataset_name: str, data_pairs_path: str, data_nonpaired_path: str, pairs: bool):
    """ Loads a full set of sentence similarities

    Args:
        dataset_name (str): name of dataset to load
        data_pairs_path (str): path to paired sentence files
        data_nonpaired_path (str): path to non-paired sentence files
        pairs (bool): load sentence pairs or a list of sentences

    Returns:
        dict: dictionary of embeddings
    """
    
    # paired data
    if pairs==True:
        try:
            sentences_dict = load_sentences(data_pairs_path+dataset_name+'.csv', pairs)
        except FileNotFoundError:
            sentences_dict = load_sentences(data_pairs_path+dataset_name+'.txt', pairs)
    # non-paired data
    elif pairs==False:
        try:
            sentences_dict = load_sentences(data_nonpaired_path+dataset_name+'.csv', pairs)
        except FileNotFoundError:
            sentences_dict = load_sentences(data_nonpaired_path+dataset_name+'.txt', pairs) 
    return sentences_dict


# Function to normalise a set of sentence embeddings
def normalise_embeddings(embeddings: np.ndarray):
    """ Normalise a set of embeddings by dividing by std and subtracting mean

    Args:
        embeddings (np.ndarray): unstandardised embeddings

    Returns:
        np.ndarray: standardised embeddings
    """
    mean_np = np.mean(embeddings, axis=0)
    std_np = np.std(embeddings, axis=1)
    new_mean_np = np.transpose(embeddings - mean_np)
    new_embeddings = np.transpose(new_mean_np/std_np)
    return new_embeddings


# Load a set of sentence embeddings
def load_embeds(path_base: str, dataset_name: str, full_sent_sim_func_mod: str):
    """ Load a set of sentence embeddings from pre-saved files.

    Args:
        path_base (str): base path to folder containing files
        dataset_name (str): name of dataset to load embeddings for
        full_sent_sim_func_mod (str): type of composition funcdtion to load embeddings for

    Returns:
        np.ndarry: array of sentence embeddings
    """
    sentences_a = np.loadtxt(path_base+dataset_name+'_a_'+full_sent_sim_func_mod+'_embeddings.txt',  delimiter=' ', dtype='float', encoding='utf-8')
    sentences_b = np.loadtxt(path_base+dataset_name+'_b_'+full_sent_sim_func_mod+'_embeddings.txt',  delimiter=' ', dtype='float', encoding='utf-8')
    all_sentence_embeds = []
    i = 0
    for i in range(sentences_a.shape[0]):
        new_entry = np.array([sentences_a[i],sentences_b[i]]).flatten()
        all_sentence_embeds.append(new_entry)
    return all_sentence_embeds


# function to convert a flattened upper triangle to a full nxn matrix
def mask_based_utri2mat(upper_triangle, ntotal):
    """ Convert a flattened upper trianngular matrix into an nxn matrix.

    Args:
        upper_triangle (np.ndarray): a flattened upper triangular matrix
        ntotal (int): the dimension of the full matrix

    Returns:
        np.ndarray: full nxn matrix
    """
    out = np.ones((ntotal,ntotal)) # output array
    mask = np.triu(np.ones((ntotal,ntotal),dtype=bool), k=1) # create mask for upper triangle
    out[mask] = upper_triangle  # upper triangular elements with mask
    out.T[mask] = upper_triangle  # lower triangular elements with transposed mask
    return out 


# function to extract name of model from file name
def extract_text_between(regex_pattern, text):
    """ Use a regex pattern to extract the model name from a filename.

    Args:
        regex_pattern (str): regex pattern
        text (str): filename 

    Returns:
        str: name of model
    """
    matches = re.findall(regex_pattern, text)
    results = []
    for match in matches:
        start_index = text.index(match)
        end_index = start_index + len(match)
        results.append(text[start_index:end_index])
    return results


# function to load sentence similarity values from file
def load_sentence_sim_values(filename):
    """ Load sentence similarity values from a file.

    Args:
        filename (str): filename

    Returns:
        np.ndarray: similarity values
    """
    try:
        with open(filename, encoding='utf-8') as file:
            values = np.array([float(line.rstrip('\n')) for line in file])
    except FileNotFoundError:
        print('could not load '+filename)
        values = []
    return values


# loop over all models to load data from the sim storage folder (for sentence similarities)
def load_model_sims(all_files, full_dataset_name, path_root, sims_path):
    """ Load sentence similarity data from a list of files for a given dataset.

    Args:
        all_files (list of str): list of files to load data from
        full_dataset_name (str): dataset to load data for
        path_root (str): root path to data files
        sims_path (str): path to where similarity files are stored

    Returns:
        (np.ndarray,np.ndarray): tuple of similarities for each model
    """
    sim_storage = {}
    sim_storage_adj = {}
    for file_name in all_files:
        
        # Regular expression pattern to match files for given model
        filename_pattern = r"^{}.*\.txt$".format(full_dataset_name)

        # Check if the filename matches the pattern
        if re.match(filename_pattern, file_name):
            print(file_name)
            basename_pattern = r"{}_(.*?)_similarities.txt".format(full_dataset_name) # extract basename to save
            base_name = extract_text_between(basename_pattern, file_name)[0]
            sim_storage[base_name] = load_sentence_sim_values(path_root+sims_path+file_name)

            # adjust for location of peak and range of similarities
            sim_median = np.median(sim_storage[base_name])
            sim_storage_adj[base_name] = (sim_storage[base_name]-sim_median)/(np.max(sim_storage[base_name])-sim_median) 
    
    return (sim_storage,sim_storage_adj)


# make adjustments to dataset name to get file to load, as some have idiosyncratic naming issues
def fix_sentence_dataset_name(dataset):
    """ Make adjustments to dataset names for loading files.

    Args:
        dataset (str): dataset filename

    Returns:
        str: model name
    """
    if dataset.split('\\')[1] == 'Fodor2023-final240' or dataset.split('\\')[1] == 'Fodor2023-final192' or dataset.split('\\')[1] == 'Fodor2023-prelim':
        full_dataset_name = dataset.split('\\')[1]
    elif dataset == 'STS3k_all_rand' or dataset == 'STS3k_all_expr_501': # Need to rename the STSk variants
        full_dataset_name = 'STS3k_all'
    elif dataset.split('\\')[2]=='stimuli_243sentences':
        full_dataset_name = dataset.split('\\')[0].split(' ')[1]+'-243_neuro'
    elif dataset.split('\\')[2]=='stimuli_384sentences':
        full_dataset_name = dataset.split('\\')[0].split(' ')[1]+'-384_neuro'
    else:
        full_dataset_name = dataset.split('\\')[0].split(' ')[1]+'_neuro'
    
    return full_dataset_name


# load your openai key from a file
def load_openai_key(key_path):
    """ Loads the OpenAI key and org id for your account from a local file

    Args:
        key_path (str): path to OpenAI key

    Returns:
        str,str: key, org_id
    """
    with open(key_path, encoding='utf-8') as file: # load openai embeddings key from file
        key = [line.rstrip('\n') for line in file]
    return key[0],key[1] # key, org


# Define dictionaries of available sentence datasets
available_pair_datasets = ['GS2011_processed', 'KS2013_processed', 'Fodor_pilot_2022', 'STS131_processed', 'SICK_relatedness',
                           'STSb_captions_test', 'STSb_forums_test', 'STSb_headlines_test', 'STSb_test', 'STS3k_all']
keys = np.arange(len(available_pair_datasets))
available_pair_datasets = dict(zip(keys, available_pair_datasets))
available_nonpaired_datasets = ['2014 Wehbe\Stimuli\\Chapter_9_sentences_final', '2017 Anderson\\Stimuli\\stimuli_final',
                                '2018 Pereira\\Stimuli\\stimuli_243sentences', '2018 Pereira\\Stimuli\\stimuli_384sentences', '2020 Alice Dataset\\Stimuli\\stimuli_sentences_final',
                                '2020 Zhang\\Stimuli\\test_sentences_final',
                                '2023 Fodor Dataset\\Fodor2023-final240','2023 Fodor Dataset\\Fodor2023-final192', '2023 Fodor Dataset\\Fodor2023-prelim']
keys = np.arange(len(available_nonpaired_datasets))
available_nonpaired_datasets = dict(zip(keys, available_nonpaired_datasets))
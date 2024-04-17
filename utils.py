import torch
import torchtext

glove = torchtext.vocab.GloVe(name="6B", dim=50, cache='.vector_cache')  # trained on Wikipedia 2014 + Gigaword 5 vectors

def euclidean_dist_sum(w_lst):
    '''
    Takes in a list of word, computes the sum of euclidean disctance of each
    word-word pair
    '''
    
    eu_dist = torch.tensor(0.0)     # initiates the euclidean distance

    for idx, word_x in enumerate(w_lst):
        if idx != len(w_lst) - 1:       # if not the last word        
            x = glove[word_x]
            for word_y in w_lst[idx+1:]:
                y = glove[word_y]
                eu_dist += torch.norm(y-x)

    return eu_dist


def cosine_sim_sum(w_lst):
    '''
    Takes in a list of word, computes the sum of euclidean disctance of each
    word-word pair
    '''
    cosine_sim = torch.tensor([0.0])     # initiates cosine similarity value

    for idx, word_x in enumerate(w_lst):
        if idx != len(w_lst) - 1:       # if not the last word        
            x = glove[word_x]
            for word_y in w_lst[idx+1:]:
                y = glove[word_y]
                cosine_sim += torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

    return cosine_sim
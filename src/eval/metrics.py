import re
from rouge import Rouge
from fuzzywuzzy import fuzz
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens

def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    chencherry = SmoothingFunction() 
    
    if gram == 1:
        weights = (1., )
    elif gram == 2:
        weights = (1. / 2., 1. / 2.)
    elif gram == 3:
        weights = (1. / 3., 1. / 3., 1. / 3.)
    elif gram == 4:
        weights = (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
    else:
        return 0.0
    
    bleu = sentence_bleu(
        [reference_tokens], 
        hypothesis_tokens, 
        weights, 
        smoothing_function=chencherry.method1
    )
    return bleu

def calculate_bleu(prediction, target, gram):
    if not prediction or not target:
        return 0.0  # Skip if empty
    return bleu_score(target, prediction, gram)

########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    return scores['rouge-l']['f']

def calculate_rouge(prediction, target):
    if not prediction or not target:
        return 0.0  # Skip if empty
    return score_rouge(target, prediction)

########################
## Accuracy (EM)
########################
def calculate_accuracy(prediction, target):
    if not prediction or not target:
        return 0.0  # Skip if empty
    return 1.0 if prediction == target else 0.0

########################
## F1-micro
########################
def f1_score(list1, list2):
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1) if len(list1) > 0 else 0
    recall = num_TP / len(list2) if len(list2) > 0 else 0
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))

def calculate_f1(prediction, target):
    if not prediction or not target:
        return 0.0  # Skip if empty
    return f1_score(target, prediction)

########################
## fuzzywuzzy
########################
def calculate_fuzz(prediction, target):
    if not prediction or not target:
        return 0.0  # Skip if empty
    return fuzz.ratio(prediction, target)

########################
## SARI
########################
def calculate_sari(inputs, prediction, target):
    sari = load("sari")
    return sari.compute(sources=[inputs], predictions=[prediction], references=[[target]])['sari']
import re
from typing import List
from decimal import Decimal, ROUND_HALF_UP

import os
import sys

import string



def normalize_number(value: str) -> Decimal:
    """Convert the string to Decimal, supporting percentages."""
    if value.endswith('%'):
        value = value.strip('%')
        decimal_value = Decimal(value) / Decimal('100')
        return decimal_value.quantize(Decimal('1.0000'), rounding=ROUND_HALF_UP)
    return Decimal(value)


def get_decimal_precision(values: List[str]) -> int:
    """Get the smallest number of decimal places in a set of references (without percentages)"""
    precisions = []
    for val in values:
        if val.endswith('%'):
            continue
        if '.' in val:
            precisions.append(len(val.split('.')[-1]))
        else:
            precisions.append(0)
    return min(precisions) if precisions else 0


def round_decimal(value: Decimal, precision: int) -> str:
    """Rounded to specified decimal places"""
    rounding_format = f'1.{"0" * precision}'
    return str(value.quantize(Decimal(rounding_format), rounding=ROUND_HALF_UP))


def is_number(val: str) -> bool:
    """Determine if it is in the form of a number or percentage"""
    val = val.strip()
    return bool(re.match(r'^-?\d+(\.\d+)?%?$', val))


def compute_em(references: List[str], predictions: List[str]) -> float:
    """Evaluate overall EM values and consider inconsistencies in the number of predicted outcomes"""
    total_score = 0.0
    total_count = 0

    for pred, ref in zip(predictions, references):
        ref_answers = [x.strip() for x in ref.split(',')]
        pred_answers = [x.strip() for x in pred.split(',')]

        match_score = 0.0
        weight = 1.0 / len(ref_answers)

        for i, r in enumerate(ref_answers):
            if i >= len(pred_answers):
                continue
            p = pred_answers[i]
            if is_number(r):
                try:
                    if r.endswith('%'):
                        norm_r = normalize_number(r)
                        norm_p = normalize_number(p)
                        if norm_r == norm_p:
                            match_score += weight
                    else:
                        ref_vals = [x for x in ref_answers if is_number(
                            x) and not x.endswith('%')]
                        precision = get_decimal_precision(ref_vals)
                        norm_r = round_decimal(
                            normalize_number(r), precision)
                        norm_p = round_decimal(
                            normalize_number(p), precision)
                        if norm_r == norm_p:
                            match_score += weight
                except:
                    continue
            else:
                if r == p:
                    match_score += weight

        total_score += match_score
        total_count += 1

    return total_score / total_count if total_count else 0.0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class QAMetric:

    def __init__(self, **kwargs):
        pass

    def prepsocess(self, references, predictions):
        '''
        Preprocess predictions and references
        '''
        processed_predictions = []
        processed_references = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            # normalize prediction and reference
            prediction = normalize_answer(prediction)
            reference = normalize_answer(reference)
            # add prediction and reference to processed list
            processed_predictions.append(prediction)
            processed_references.append(reference)
        predictions = processed_predictions
        references = processed_references
        return references, predictions

    def compute(self, references, predictions):
        '''
        Support Mtrics: EM, ROUGE-L
        '''
        metric_scores = {}
        references, predictions = self.prepsocess(references, predictions)

        sys.setrecursionlimit(8735 * 2080 + 10)
        # calculate F1,EM, ROUGE-L, SacreBLEU, Meteor
        em_score = compute_em(references=references, predictions=predictions)

        metric_scores = {
            'EM': round(em_score*100, 2),
        }

        return metric_scores
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from collections import defaultdict
import json
import re
import copy

def vectorize_query_local(query_str, min_max, encoders):
    query_str = query_str.replace("NULL", "-1").replace("IS NOT", "<>")
    total_columns = len(min_max)
    vector = np.zeros(total_columns*4)
    predicates = query_str.split("WHERE", maxsplit=1)[1]
    operators = {
        "=": [0,1,0],
        ">": [0,0,1],
        "<": [1,0,0],
        "<=": [1,1,0],
        ">=": [0,1,1],
        "<>": [1,0,1],
        "IS": [0,1,0]
    }
    
    for exp in predicates.split("AND"):
        exp = exp.strip()
        if " " in exp:
            pred, op, value = exp.split(" ")
        else:
            continue
        if pred in encoders.keys():
            value = encoders[pred].transform([value.replace("'", "")])[0]
        else:
            value = min(max(min_max[pred][0], float(value)), min_max[pred][1])
        idx = list(sorted(min_max.keys())).index(pred)
        vector[idx*4:idx*4+3] = operators[op]
        vector[idx*4+3] = (value-min_max[pred][0]+min_max[pred][2])/ \
                          (min_max[pred][1]-min_max[pred][0]+min_max[pred][2])
    
    return vector

def vectorize_query_range(query_str, min_max, encoders):
    query_str = query_str.replace("NULL", "-1").replace("IS NOT", "<>")
    total_columns = len(min_max)
    vector = np.zeros(total_columns*6)
    predicates = query_str.split("WHERE", maxsplit=1)[1]
    operators = {
        "=": [0,1],
        ">": [1,0],
        ">=": [1,1],
        "<": [1,0],
        "<=": [1,1],
        "<>": [1,0],
        "IS": [0,1],
        "": [0,0]
    }
    
    #collect bounds
    bounds = defaultdict(list)
    for exp in predicates.split("AND"):
        exp = exp.strip()
        #print(exp)
        if " " in exp:
            exp = exp.split(" ")
        else:
            continue
        if exp[0] in encoders.keys():
            exp[-1] = encoders[exp[0]].transform([exp[-1]])[0]
        else:
            exp[-1] = min(max(min_max[exp[0]][0], float(exp[-1])), min_max[exp[0]][1])
        bounds[exp[0]].append(exp[1:])
    #print(bounds)       
    
    for pred, limits in bounds.items():
        # extend incomplete bounds and single bounds <>, =
        if len(limits) < 2:
            if limits[0][0] == "<>" or limits[0][0] == "=":
                limits.append(limits[0])
            elif ">" in limits[0][0]:
                limits.append(["<=", min_max[pred][1]])
            elif "<" in limits[0][0]:
                limits.insert(0, [">=", min_max[pred][0]])
        
        offset = 0
        # only upper and lower -> offset = 0 then offset = 3
        for op, bound in limits:
            idx = list(sorted(min_max.keys())).index(pred)
            
            vector[idx*6+offset:idx*6+offset+2] = operators[op]
            if bound is None:
                vector[idx*6+offset+2] = 0
            else:
                vector[idx*6+offset+2] = (bound-min_max[pred][0]+min_max[pred][2])/ \
                                         (min_max[pred][1]-min_max[pred][0]+min_max[pred][2])
            
            offset += 3
    
    return vector

# helper function
def prepare_data_structures(min_max, max_bucket_count):
    feature_vectors = dict() # dict of floats by attribute
    atomar_buckets = dict() # dict of booleans by attribute
    not_values = dict() # dict of list by attribute

    for attr, domain in min_max.items():
        domainrange = domain[1] - domain[0]
        if max_bucket_count < domainrange:
            bucket_count = max_bucket_count
            atomar_buckets[attr] = False
        else:
            bucket_count = domainrange
            atomar_buckets[attr] = True
        feature_vectors[attr] = np.ones(bucket_count + 1) # last one is for covered ratio
        bounds = {attr : list(vals) for attr, vals in min_max.items()}
        not_values[attr] = []

    return feature_vectors, atomar_buckets, bounds, not_values


#helper function
def add_simplepred_to_featurevec(attr_feature_vec, val_bucket_idx, attr, op, val, min_max, atomar_buckets, bounds, not_values):
    if op == "=" or op == "IS":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][0] = val
        bounds[attr][1] = val+1
    elif op == ">":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 0 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        bounds[attr][0] = max(bounds[attr][0], min(val+1, min_max[attr][1]))
    elif op == "<":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][1] = min(bounds[attr][1], max(val-1, min_max[attr][0]))
    elif op == "<=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][1] = min(bounds[attr][1], val)
    elif op == ">=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        bounds[attr][0] = max(bounds[attr][0], val)
    elif op == "<>" or op == "!=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 0 if atomar_buckets[attr] else 0.5
        not_values[attr].append(val)
    else:
        raise SystemExit("Unknown operator", op)
    
    return attr_feature_vec


# helper function
def add_compoundpred_to_featurevec (simple_predicates, vec, min_max, atomar_buckets, bounds, not_values, encoders):
    #print("simple_predicates:", simple_predicates)
    for exp in simple_predicates.split("AND"):
        exp = exp.strip()
        # assert exp has form: attribute operator value
        #print("exp:", exp)
        attr, op, val = exp.split(" ")
        if attr in encoders.keys():
            val = encoders[attr].transform([val.replace("'", "")])[0]
        else:
            val = min(max(min_max[attr][0], float(val)), min_max[attr][1])
        #print("attr-op-val", attr, op, val)
        domainrange = min_max[attr][1] - min_max[attr][0] + 1
        positionval = val - min_max[attr][0]
        # k = positionval / domainrange in [0,1), floor(k * len(vector)) gives number [0, len(vector)-1]
        val_bucket_idx = int(float(positionval) / domainrange * len(vec))
        add_simplepred_to_featurevec(vec, val_bucket_idx, attr, op, val, min_max, 
                                     atomar_buckets, bounds, not_values)
    return vec

def vectorize_attribute_domains_complex_query(query_str, min_max, encoders, max_bucket_count = 128):
    query_str = query_str.replace("NULL", "-1").replace("IS NOT", "<>")
    feature_vectors, atomar_buckets, bounds, not_values = prepare_data_structures(min_max, max_bucket_count)

    complex_query = query_str.split("WHERE", maxsplit=1)[1]
    compound_predicates = re.findall('\(.*?\)', complex_query)
    compound_predicates = [cp.strip("()") for cp in compound_predicates]
    for cp in compound_predicates:
        attr = cp[0]
        disjuncts = re.split("OR", cp)
        mergedvec = np.zeros(len(feature_vectors[attr]))
        for disjunct in disjuncts: # each disjunct is conjunction of simple predicates
            dis_vec = add_compoundpred_to_featurevec(disjunct, np.copy(feature_vectors[attr]), min_max,
                                                     atomar_buckets,
                                                     bounds,
                                                     not_values,
                                                     encoders)
            mergedvec = np.maximum(mergedvec, dis_vec)
        #ENDFOR
        mergedvec[-1] = sum(mergedvec[0:-1]) / (len(mergedvec)-1) # approximate covered ratio
        feature_vectors[attr] = mergedvec
    #ENDFOR

    totalfeaturevec = np.concatenate(list(feature_vectors.values()))
    return totalfeaturevec
    


def vectorize_attribute_domains_no_disjunctions(query_str, min_max, encoders, max_bucket_count = 128):
    query_str = query_str.replace("NULL", "-1").replace("IS NOT", "<>")
    feature_vectors, atomar_buckets, bounds, not_values = prepare_data_structures(min_max, max_bucket_count)

    complex_query = query_str.split("WHERE", maxsplit=1)[1]
    simple_predicates = complex_query.split("AND")
    for exp in simple_predicates:
        exp = exp.strip()
        if not " " in exp: continue  # join predicate
        attr, op, val = exp.split(" ")
        if attr in encoders.keys():
            val = encoders[attr].transform([val.replace("'", "")])[0]
        else:
            val = min(max(min_max[attr][0], float(val)), min_max[attr][1])
        attr_feature_vec = feature_vectors[attr]
        domainrange = min_max[attr][1] - min_max[attr][0] + 1
        positionval = val - min_max[attr][0]
        # k = positionval / domainrange in [0,1), floor(k * len(vector)) gives number [0, len(vector)-1]
        val_bucket_idx = int(float(positionval) / domainrange * len(feature_vectors[attr]))
        add_simplepred_to_featurevec(attr_feature_vec, val_bucket_idx, attr, op, val, 
                                     min_max, atomar_buckets, bounds, not_values)

    attributes = min_max.keys()
    for attr in attributes:
        # set covered domain ratio
        domainrange = min_max[attr][1] - min_max[attr][0]
        queryrange = bounds[attr][1] - bounds[attr][0]
        notsum = sum( [1 for x in not_values[attr] if bounds[attr][0] <= x <= bounds[attr][1]])
        queryrange = max(queryrange - notsum,  0)
        feature_vectors[attr][-1] = queryrange / domainrange

    totalfeaturevec = np.concatenate(list(feature_vectors.values()))
    return totalfeaturevec

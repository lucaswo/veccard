import random

# Helper function.
# Returns either a string that represents a conjunction of predicates for attribute "attr" in SQL syntax.
# Or, with probability "null_pred_prob", returns an "IS NULL" predicate.
def get_conjunction_for_attr(attr,
                             min_max,
                             encoders,
                             max_not_equal_preds,
                             null_pred_prob,
                             neq_preds_in_range_pred,
                             neq_symbol_with_spaces = " <> "):
    # nested helper function
    # returns random value in domain of "attr"
    def get_value(attr, min_max, encoders, value_bounds=None):
            if not value_bounds:
                # currently only supports integer columns
                value_bounds = (min_max[attr][0], min_max[attr][1])
            if attr in encoders.keys():
                print("ToDo: Ask Lucas to explain code in 'if' block (What is 'classes_' collection member?)")
                upper = len(encoders[attr].classes_) - 1
                random_idx = random.randint(0, upper)
                val = encoders[attr].classes_[random_idx]
            else:
                #random.randrange generates integers. Version with floats needed?
                val = random.randrange(value_bounds[0],  value_bounds[1] + 1,  min_max[attr][2])
            return val

    if random.uniform(0.0, 1.0) <= null_pred_prob:
        return attr + " IS NULL"

    preds = []
    # generate random range predicate
    lower = get_value(attr, min_max, encoders)
    upper = get_value(attr, min_max, encoders)
    if lower > upper:
        lower, upper = upper, lower
    rangepred = attr + " >= " + str(lower) + " AND " + attr + " <= " + str(upper)
    preds.append(rangepred)

    # generate up to "max_not_equal_preds" many not-equal predicates
    not_equal_count = random.randint(0, max_not_equal_preds)
    if not_equal_count > 0:
        if neq_preds_in_range_pred:
            not_equal_vals = [get_value(attr, min_max, encoders, (lower, upper)) for i in range(0, not_equal_count)]
        else:
            not_equal_vals = [get_value(attr, min_max, encoders) for i in range(0, not_equal_count)]
        neq_preds = " AND ".join(attr + neq_symbol_with_spaces + str(v) for v in not_equal_vals)
        preds.append(neq_preds)

    ## with probability "null_pred_prob", generate "IS NOT NULL" predicate
    ## Note: If range-predicate is satisfied, then attribute value must be not null
    #if random.uniform(0.0, 1.0) <= null_pred_prob:
    #    not_null_pred = attr + " IS NOT NULL"
    #    preds.append(not_null_pred)

    conjunctive_pred_string = " AND ".join(p for p in preds)
    return conjunctive_pred_string




def generate_queries_conjunctive(cur, n_queries, min_max, encoders,
                                 vectorize_query_f = None,
                                 max_not_equal_preds = 5,
                                 null_pred_prob = 0.0,
                                 neq_preds_in_range_pred = True):
    accepted_queries_set = set()
    taboo_queries_set = set()
    queries_list = list()  # part of returned result
    cardinalities_list = list()  # part of returned result
    attributes = list(min_max.keys())

    while len(accepted_queries_set) < n_queries:
        attribute_count = random.randint(1, len(attributes))
        selected_attrs = random.sample(attributes, attribute_count)
        preds = []

        # create conjunctively connected predicates for "attr"
        for attr in selected_attrs:
            preds.append(get_conjunction_for_attr(attr,
                                                  min_max,
                                                  encoders,
                                                  max_not_equal_preds,
                                                  null_pred_prob,
                                                  neq_preds_in_range_pred))
        #END_FOR

        sql_str = "SELECT count(*) FROM " + \
                  config["view_name"] + " WHERE " + \
                  " AND ".join(p for p in preds)
        if sql_str in accepted_queries_set or sql_str in taboo_queries_set:
            continue
        cur.execute(sql_str)
        query_result_card = cur.getchone()[0]
        if query_result_card == 0:
            taboo_queries_set.add(sql_str)
            continue
        accepted_queries_set.add(sql_str)
        queries_list.append(sql_str)
        cardinalities_list.append(query_result_card)
    #END_WHILE

    vectors = []
    if vectorize_query_f:
        vectors = [vectorize_query_f(q, min_max, encoders) for q in queries_list]
    
    return queries_list, vectors, cardinalities_list




def generate_queries_complex(cur, n_queries, min_max, encoders,
                                 vectorize_query_f = None,
                                 max_disjuncts = 3,
                                 max_not_equal_preds = 5,
                                 null_pred_prob = 0.0,
                                 neq_preds_in_range_pred = True):
    accepted_queries_set = set()
    taboo_queries_set = set()
    queries_list = list()  # part of returned result
    cardinalities_list = list()  # part of returned result
    attributes = list(min_max.keys())

    while len(accepted_queries_set) < n_queries:
        attribute_count = random.randint(1, len(attributes))
        selected_attrs = random.sample(attributes, attribute_count)
        complex_query = []  # total disjunctive predicate / disjunction of all compund predicates
        # create compund predicate for "attr", e.g.: "(p1 and p2 or p3 and p4 and p5 or p6)"
        for attr in selected_attrs:
            disjunction_count = random.randint(1, max_disjuncts)
            curr_attr_compound_pred = []
            for i in range(0, disjunction_count):
                curr_attr_compound_pred.append(get_conjunction_for_attr(attr,
                                                                            min_max,
                                                                            encoders,
                                                                            max_not_equal_preds,
                                                                            null_pred_prob,
                                                                            neq_preds_in_range_pred))
            #END_FOR
            complex_query.append("(" + " OR ".join(p for p in curr_attr_compound_pred) + ")")
        #END_FOR
        sql_str = "SELECT count(*) FROM " + \
                  config["view_name"] + " WHERE " + \
                  " AND ".join(p for p in complex_query)
        if sql_str in accepted_queries_set or sql_str in taboo_queries_set:
            continue
        cur.execute(sql_str)
        query_result_card = cur.getchone()[0]
        if query_result_card == 0:
            taboo_queries_set.add(sql_str)
            continue
        accepted_queries_set.add(sql_str)
        queries_list.append(sql_str)
        cardinalities_list.append(query_result_card)
    #END_WHILE

    vectors = []
    if vectorize_query_f:
        vectors = [vectorize_query_f(q, min_max, encoders) for q in queries_list]

    return queries_list, vectors, cardinalities_list




min_max = {
    "A" : (0, 100, 1),
    "B" : (2, 777, 1),
    "C" : (7, 40, 1)
}

encoders = {}

config = {"view_name" : "TABLENAME"}

class SqlHandleMock:
    def execute(x):
        None
    def getchone():
        return [77]

mock_cur = SqlHandleMock
mock_vectorizer = lambda x, y, z: [0,1,1]

Q, V, C = generate_queries_conjunctive(mock_cur, 5, min_max, encoders, mock_vectorizer)
print("Arbitrary conjunction queries:")
for q in Q:
    print(q)

print("\n\n\n")
Q, V, C = generate_queries_complex(mock_cur, 5, min_max, encoders,
                                   null_pred_prob=0.2,
                                   neq_preds_in_range_pred=False)
print("Complex queries:")
for q in Q:
    print(q)


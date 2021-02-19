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

def generate_queries_local(cur, n_queries, min_max, encoders, tables, join_ids, cube=False, 
                           cube_columns=None, trad_est=False):
    SQL_set = set()
    SQL0_set = set()
    SQL = []
    cardinalities = []
    estimates = []
    if len(tables) > 1:
        sql_body = """SELECT * FROM {} WHERE {} AND {};""".format(",".join(tables), "=".join(join_ids), "{}")
    else:
        sql_body = """SELECT * FROM {} WHERE {};""".format(",".join(tables), "{}")
    if cube:
        sql_body_count_cube = """SELECT COALESCE(SUM(count), 0) FROM cube{} WHERE {};""".format(cube, "{}")
    sql_body_count = """SELECT count(*) FROM tmpview{} WHERE {};""".format(cube, "{}")
    
    total_columns = len(min_max)
    vectors = np.ndarray((n_queries, total_columns*4))
    columns = list(min_max.keys())
    count = 0
    while len(SQL) < n_queries:
        num_of_predictates = np.random.choice(range(1,total_columns+1))#, 
        #                                      p=[0.3, 0.3, 0.2, 0.1, 0.025, 0.025, 0.025, 0.025])
        selected_predicates = np.random.choice(range(total_columns), size=num_of_predictates, replace=False)
        selected_predicates = [columns[i] for i in selected_predicates]
        
        selected_values = []
        for pred in selected_predicates:
            if pred in encoders.keys():
                sel = np.random.randint(len(encoders[pred].classes_))
                sel = encoders[pred].classes_[sel]
            else:
                choices = np.arange(min_max[pred][0], min_max[pred][1]+1, min_max[pred][2])
                sel = np.random.choice(choices)
            selected_values.append(sel)
        
        #[[0,0,1], [0,1,0], [1,0,0], [1,0,1], [0,1,1], [1,1,0]]
        # <>=
        selected_operators = np.random.choice(["=", ">", "<", "<=", ">=", "!="], size=num_of_predictates)
        #selected_operators = np.random.choice(["=", ">", "<"], size=num_of_predictates)
        #selected_operators = ["=" if "id" not in sp else np.random.choice(["=", ">", "<", "<=", ">=", "!="]) 
        #                      for sp in selected_predicates]
        #selected_operators = [np.random.choice(["IS", "IS NOT"]) if selected_values[i] == "-1" 
        #                      or selected_values == -1 else x for i,x in enumerate(selected_operators)]
        selected_operators = ["IS" if selected_values[i] == "-1" or selected_values == -1 else x 
                              for i,x in enumerate(selected_operators)]
        
        predicates_str = " AND ".join([" ".join([str(p), str(o), str(v) if not isinstance(v,str) or v == "-1"
                                                      else "'{}'".format(v)]) for p,o,v in zip(selected_predicates,
                                                                                               selected_operators, 
                                                                                               selected_values)])
        sql = sql_body.format(predicates_str)
        #print(selected_predicates, cube_columns)
        if not set(selected_predicates) - set(cube_columns):
            sql_count = sql_body_count_cube.format(predicates_str)
        else:
            sql_count = sql_body_count.format(predicates_str)

        check_len = len(SQL_set)
        #sql = sql.replace("-1", "NULL")
        #sql_count = sql_count.replace("-1", "NULL")
        SQL_set.add(sql)
        if check_len != len(SQL_set) and sql not in SQL0_set:
            cur.execute(sql_count)
            card = int(cur.fetchone()[0])
            
            if card > 0:
                SQL.append(sql)
                cardinalities.append(card)
                vectors[len(SQL)-1] = vectorize_query(sql_count, min_max, encoders)
                
                if trad_est:
                    cur.execute("EXPLAIN {}".format(sql))
                    est = cur.fetchone()[0]
                    est = int(re.findall(r"rows=(\d+)", est)[0])
                    estimates.append(est)
                
                #if not len(SQL) % 1000:
                #    print(len(SQL))
            else:
                SQL0_set.add(sql)
                SQL_set.remove(sql)
            
        count += 1
    
    print(count)
    if trad_est:
        return SQL, vectors, cardinalities, estimates
    else:
        return SQL, vectors, cardinalities

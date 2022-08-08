def rules_difference(list_a, list_b):
    a = set(map(tuple, list_a))
    b = set(map(tuple, list_b))
    list_d = a - b
    ordered = sorted(list(list_d), key = len)
    return ordered

def rules_intersection(list_a, list_b):
    a = set(map(tuple, list_a))
    b = set(map(tuple, list_b))
    list_i = a & b
    return list(list_i)

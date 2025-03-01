#function creating the relevance matrix.
def creating_relevance_matrix(query_lab, test_lab):
  a = len(query_lab)
  b = len(test_lab)
  relevance_matrix = np.zeros((a,b), dtype=int)
  for i in range(len(query_lab)): 
    lab_q = query_lab[i]
    for j in range(len(test_lab)):
      lab_test = test_lab[j]
      if lab_q == lab_test:
        relevance_matrix[i,j] = 1
  return relevance_matrix

#function calculating the precision for one query
def calculate_precision_for_kth(top_indices, relevant_indices):
  precisions = []
  relevant = 0
  for i in range(len(top_indices)):
    index = top_indices[i]
    if index in relevant_indices:
      relevant += 1
      precisions.append(relevant/(i+1))
  if precisions == []:
    return [0]
  else:
    return precisions

#function calculating the mean Average precision
def calculating_map(query_lab, test_lab, relevance_matrix, test_indices):
  ap_list = []
  
  for query_idx in range(len(query_lab)):
    top_indices = test_indices[query_idx]
    relevant_indices = np.where(relevance_matrix[query_index] == 1)[0]
    precisions = calculate_precision_for_kth(top_indices, relevant_indices)
    ap_list.append(np.mean(precisions))
  return np.mean(ap_list)

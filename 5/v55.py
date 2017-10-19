def create_page_rank_markov_chain(links, damping_factor=0.15):
    ''' По веб-графу со списком ребер links строит матрицу 
    переходных вероятностей соответствующей марковской цепи.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        damping_factor --- вероятность перехода не по ссылке (float);
        
        Возвращает prob_matrix --- numpy.matrix, shape=(|V|, |V|).
    '''

    links = np.array(links)
    N = links.max() + 1  # Число веб-страниц
    
    prob_matrix = np.ones((N, N))
    # без переходов по ссылкам
    prob_matrix *= 1/N
    
    #каждый i-ый элемент - список исходящих ребер из i
    edges = [[] for i in np.arange(N)]
    for e in links:
        edges[e[0]].append(e[1])
    
    # добавим переходы
    for i in np.arange(len(edges)):
        M = len(edges[i])
        if(M>0):
            prob_matrix[i, :] *= damping_factor
            for j in edges[i]:
                prob_matrix[i, j] += (1-damping_factor)/M
    
    return np.matrix(prob_matrix)


def page_rank(links, start_distribution, damping_factor=0.15, 
              tolerance=10 ** (-7), return_trace=False):
    ''' Вычисляет веса PageRank для веб-графа со списком ребер links 
    степенным методом, начиная с начального распределения start_distribution, 
    доводя до сходимости с точностью tolerance.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        start_distribution --- вектор размерности |V| в формате numpy.array;
        damping_factor --- вероятность перехода не по ссылке (float);
        tolerance --- точность вычисления предельного распределения;
        return_trace --- если указана, то возвращает список распределений во 
                            все моменты времени до сходимости
    
        Возвращает:
        1). если return_trace == False, то возвращает distribution --- 
        приближение предельного распределения цепи,
        которое соответствует весам PageRank.
        Имеет тип numpy.array размерности |V|.
        2). если return_trace == True, то возвращает также trace ---
        список распределений во все моменты времени до сходимости. 
        Имеет тип numpy.array размерности 
        (количество итераций) на |V|.
    '''
    
    prob_matrix = create_page_rank_markov_chain(links, 
                                                damping_factor=damping_factor)
    distribution = np.matrix(start_distribution)
    #в trace  логично добавлять и начальное, и конечное распределение, но что имеется в виду в задании не ясно - добавлю все
    trace = []
    while(True):
        trace.append(distribution)
        distribution = np.dot(distribution, prob_matrix)
        if (np.linalg.norm(distribution - trace[-1]) < tolerance):
            break
        
    trace.append(distribution)
    
    if return_trace:
        return np.array(distribution).ravel(), np.array(trace)
    else:
        return np.array(distribution).ravel()
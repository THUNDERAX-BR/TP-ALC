import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num = len(corpus)
    prob = {}
    links = corpus[page]

    #pagina sem links
    if len(links) == 0:
        for p in corpus:
            prob[p] = 1 / num

        return prob
    
    #pelo menos um link
    salto = (1 - damping_factor) / num
    link = damping_factor / len(links)

    for p in corpus:
        prob[p] = salto
        if p in links:
            prob[p] += link
    
    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}

    for p in corpus:
        pagerank[p] = 0

    #escolhe o primeiro aleatoriamente
    atual = random.choice(list(corpus.keys()))

    #n samples
    for i in range(n):
        pagerank[atual] += 1

        transicao = transition_model(corpus, atual, damping_factor)
        pages = list(transicao.keys())
        probs = list(transicao.values())

        #escolhe o proximo ponderadamente de acordo com transition_model
        atual = random.choices(pages, weights=probs, k=1)[0]
    
    #normaliza para uma probabilidade (somatorio ~= 1)
    for p in pagerank:
        pagerank[p] /= n
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    n = len(pages)

    #associando paginas a indices
    index = {}
    for i, p in enumerate(pages):
        index[p] = i

    #matriz L
    L = np.zeros((n,n))

    #matriz M inversa
    M = np.zeros((n,n))

    #preenche as matrizes
    for j, p in enumerate(pages):
        links = corpus[p]

        #pagina sem links
        if len(links) == 0:
            links = set(pages)

        M[j][j] = 1 / len(links)

        for page in links:
            i = index[page]
            L[i][j] = 1
        
    #mariz E
    E = np.ones((n,n))

    #matriz G
    G = ((1 - damping_factor) / n) * E + damping_factor * (L @ M)

    p = np.full((n, 1), 1 / n)
    while True:

        prox_p = G @ p

        #encerra quando a diferenca e pequena
        if np.all(np.abs(prox_p - p) <= 0.001):
            p = prox_p
            break

        p = prox_p
    
    pagerank = {}
    for i, page in enumerate(pages):
        pagerank[page] = float(p[i][0])

    return pagerank


if __name__ == "__main__":
    main()

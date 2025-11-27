# # # bio_algorithms/genetic_optimizer.py

# # import random
# # from typing import List, Tuple
# # from sentence_transformers import SentenceTransformer, util

# # class GeneticOptimizer:
# #     """
# #     Genetic Algorithm to choose the best summary among multiple candidates.
# #     Fitness is based on semantic similarity + keyword coverage.
# #     """

# #     def __init__(self, population_size: int = 5, generations: int = 5, mutation_rate: float = 0.2):
# #         self.population_size = population_size
# #         self.generations = generations
# #         self.mutation_rate = mutation_rate
# #         # Model to compute semantic similarity between transcript and summary
# #         self.model = SentenceTransformer("all-MiniLM-L6-v2")

# #     def _fitness(self, summary: str, transcript: str) -> float:
# #         # 1) semantic similarity
# #         emb_sum = self.model.encode(summary, convert_to_tensor=True)
# #         emb_tr = self.model.encode(transcript, convert_to_tensor=True)
# #         sim = util.cos_sim(emb_sum, emb_tr).item()  # between -1 and 1

# #         # 2) keyword presence (action, decision, deadline, risk)
# #         keywords = ["action", "task", "deadline", "decide", "decision", "risk"]
# #         keyword_score = sum(1 for k in keywords if k.lower() in summary.lower())

# #         # weighted sum
# #         return sim * 0.8 + keyword_score * 0.2

# #     def _crossover(self, s1: str, s2: str) -> str:
# #         """
# #         Simple crossover: take first half of s1 and second half of s2.
# #         Works because summaries are plain text / bullet points.
# #         """
# #         mid1 = len(s1) // 2
# #         mid2 = len(s2) // 2
# #         return (s1[:mid1].strip() + " " + s2[mid2:].strip()).strip()

# #     def _mutate(self, s: str) -> str:
# #         """
# #         Simple mutation: occasionally rewrite some generic phrases.
# #         (You can expand this later.)
# #         """
# #         if random.random() > self.mutation_rate:
# #             return s

# #         replacements = {
# #             "we will": "the team will",
# #             "we should": "the team should",
# #             "needs to": "is required to",
# #         }
# #         for old, new in replacements.items():
# #             if old in s.lower():
# #                 return s.replace(old, new)
# #         return s

# #     def evolve(self, candidates: List[str], transcript: str) -> Tuple[str, float]:
# #         """
# #         Run GA over a list of candidate summaries.
# #         Returns: (best_summary_text, best_fitness_score)
# #         """
# #         # start with given candidates as initial population
# #         population = candidates[:]

# #         for _ in range(self.generations):
# #             # rank by fitness
# #             scored = [(s, self._fitness(s, transcript)) for s in population]
# #             scored.sort(key=lambda x: x[1], reverse=True)
# #             # keep top 2 as parents
# #             parents = [scored[0][0], scored[1][0]]

# #             # generate a child
# #             child = self._crossover(parents[0], parents[1])
# #             child = self._mutate(child)
# #             population.append(child)

# #         # final best
# #         final_scored = [(s, self._fitness(s, transcript)) for s in population]
# #         best_summary, best_score = max(final_scored, key=lambda x: x[1])
# #         return best_summary, best_score



# # bio_algorithms/genetic_optimizer.py

# import random
# from typing import List, Tuple
# from sentence_transformers import SentenceTransformer, util

# class GeneticOptimizer:
#     """
#     Genetic Algorithm to choose the best summary among multiple candidates.
#     Fitness is based on semantic similarity + keyword coverage.
#     """

#     def __init__(self, population_size: int = 5, generations: int = 5, mutation_rate: float = 0.2):
#         self.population_size = population_size
#         self.generations = generations
#         self.mutation_rate = mutation_rate
#         # Model to compute semantic similarity between transcript and summary
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#     def _fitness(self, summary: str, transcript: str) -> float:
#         # 1) semantic similarity
#         emb_sum = self.model.encode(summary, convert_to_tensor=True)
#         emb_tr = self.model.encode(transcript, convert_to_tensor=True)
#         sim = util.cos_sim(emb_sum, emb_tr).item()  # between -1 and 1

#         # 2) keyword presence (action, decision, deadline, risk)
#         keywords = ["action", "task", "deadline", "decide", "decision", "risk"]
#         keyword_score = sum(1 for k in keywords if k.lower() in summary.lower())

#         # weighted sum
#         return sim * 0.8 + keyword_score * 0.2

#     def _crossover(self, s1: str, s2: str) -> str:
#         """
#         Simple crossover: take first half of s1 and second half of s2.
#         Works because summaries are plain text / bullet points.
#         """
#         mid1 = len(s1) // 2
#         mid2 = len(s2) // 2
#         return (s1[:mid1].strip() + " " + s2[mid2:].strip()).strip()

#     def _mutate(self, s: str) -> str:
#         """
#         Simple mutation: occasionally rewrite some generic phrases.
#         (You can expand this later.)
#         """
#         if random.random() > self.mutation_rate:
#             return s

#         replacements = {
#             "we will": "the team will",
#             "we should": "the team should",
#             "needs to": "is required to",
#         }
#         for old, new in replacements.items():
#             if old in s.lower():
#                 return s.replace(old, new)
#         return s

#     def evolve(self, candidates: List[str], transcript: str) -> Tuple[str, float]:
#         """
#         Run GA over a list of candidate summaries.
#         Returns: (best_summary_text, best_fitness_score)
#         """
#         # start with given candidates as initial population
#         population = candidates[:]

#         for _ in range(self.generations):
#             # rank by fitness
#             scored = [(s, self._fitness(s, transcript)) for s in population]
#             scored.sort(key=lambda x: x[1], reverse=True)
#             # keep top 2 as parents
#             parents = [scored[0][0], scored[1][0]]

#             # generate a child
#             child = self._crossover(parents[0], parents[1])
#             child = self._mutate(child)
#             population.append(child)

#         # final best
#         final_scored = [(s, self._fitness(s, transcript)) for s in population]
#         best_summary, best_score = max(final_scored, key=lambda x: x[1])
#         return best_summary, best_score


# # bio_algorithms/genetic_optimizer.py

# import random
# from typing import List, Tuple
# from sentence_transformers import SentenceTransformer, util

# class GeneticOptimizer:
#     """
#     Genetic Algorithm to choose the best summary among multiple candidates.
#     Fitness is based on semantic similarity + keyword coverage.
#     """

#     def __init__(self, population_size: int = 5, generations: int = 5, mutation_rate: float = 0.2):
#         self.population_size = population_size
#         self.generations = generations
#         self.mutation_rate = mutation_rate

#         # Model to compute semantic similarity between transcript and summary
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#         # 🆕 store best score each generation
#         self.history = []

#     def _fitness(self, summary: str, transcript: str) -> float:
#         emb_sum = self.model.encode(summary, convert_to_tensor=True)
#         emb_tr = self.model.encode(transcript, convert_to_tensor=True)
#         sim = util.cos_sim(emb_sum, emb_tr).item()

#         keywords = ["action", "task", "deadline", "decide", "decision", "risk"]
#         keyword_score = sum(1 for k in keywords if k.lower() in summary.lower())

#         return sim * 0.8 + keyword_score * 0.2

#     def _crossover(self, s1: str, s2: str) -> str:
#         mid1 = len(s1) // 2
#         mid2 = len(s2) // 2
#         return (s1[:mid1].strip() + " " + s2[mid2:].strip()).strip()

#     def _mutate(self, s: str) -> str:
#         if random.random() > self.mutation_rate:
#             return s

#         replacements = {
#             "we will": "the team will",
#             "we should": "the team should",
#             "needs to": "is required to",
#         }
#         for old, new in replacements.items():
#             if old in s.lower():
#                 return s.replace(old, new)
#         return s

#     def evolve(self, candidates: List[str], transcript: str) -> Tuple[str, float]:
#         population = candidates[:]

#         for _ in range(self.generations):
#             scored = [(s, self._fitness(s, transcript)) for s in population]
#             scored.sort(key=lambda x: x[1], reverse=True)

#             # 🆕 track best score for convergence chart
#             self.history.append(scored[0][1])

#             parents = [scored[0][0], scored[1][0]]
#             child = self._crossover(parents[0], parents[1])
#             child = self._mutate(child)
#             population.append(child)

#         final_scored = [(s, self._fitness(s, transcript)) for s in population]
#         best_summary, best_score = max(final_scored, key=lambda x: x[1])
#         return best_summary, best_score





# bio_algorithms/genetic_optimizer.py

import random
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util

class GeneticOptimizer:
    """
    Improved Genetic Algorithm with fitness tracking, stronger mutation,
    and better exploration for visible convergence trend.
    """

    def __init__(self, population_size=6, generations=10, mutation_rate=0.25):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # semantic model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # 🔥 NEW: track fitness score across generations
        self.history = []

    def _fitness(self, summary: str, transcript: str) -> float:
        """Compute fitness based on semantic similarity + keyword presence."""
        
        emb_sum = self.model.encode(summary, convert_to_tensor=True)
        emb_tr = self.model.encode(transcript, convert_to_tensor=True)
        sim = util.cos_sim(emb_sum, emb_tr).item()

        keywords = ["action", "task", "deadline", "meeting", "plan", "next step", "risk", "decision"]
        keyword_score = sum(1 for k in keywords if k.lower() in summary.lower())

        return sim * 0.8 + keyword_score * 0.2

    def _crossover(self, s1: str, s2: str) -> str:
        """Mix half from first parent and half from second."""
        mid1 = len(s1) // 2
        mid2 = len(s2) // 2
        return (s1[:mid1].strip() + " " + s2[mid2:].strip()).strip()

    def _mutate(self, s: str) -> str:
        """🔧 Strong mutation: shuffle, shorten or expand text."""
        if random.random() > self.mutation_rate:
            return s

        words = s.split()

        if len(words) > 12:
            random.shuffle(words)

        # random shorten or keep length variation
        cutoff = max(10, len(words) - random.randint(5, 20))
        new_sentence = " ".join(words[:cutoff])

        return new_sentence.strip()

    def evolve(self, candidates: List[str], transcript: str) -> Tuple[str, float]:
        """Run Genetic Algorithm on given candidate summaries."""
        
        population = candidates[:]

        for gen in range(self.generations):
            scored = [(s, self._fitness(s, transcript)) for s in population]
            scored.sort(key=lambda x: x[1], reverse=True)

            # log fitness for convergence graph
            best_score = scored[0][1]
            self.history.append(best_score)

            parents = [scored[0][0], scored[1][0]]

            # create new child
            child = self._crossover(parents[0], parents[1])
            child = self._mutate(child)
            population.append(child)

        final_scored = [(s, self._fitness(s, transcript)) for s in population]
        best_summary, best_score = max(final_scored, key=lambda x: x[1])

        return best_summary, best_score

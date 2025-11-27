# import os
# import json
# from typing import Dict
# import google.generativeai as genai

# # OpenAI (optional fallback)
# try:
#     from openai import OpenAI
#     _has_openai = True
# except Exception:
#     _has_openai = False

# SYSTEM_PROMPT = (
#     "You are an AI meeting summarizer.\n"
#     "1) Summarize the meeting into 3–5 bullet points.\n"
#     "2) Extract key action items.\n"
#     "3) List any decisions made.\n"
#     "4) Identify risks.\n"
#     "5) Provide a fluent translation of the summary in the target language.\n"
#     "6) Perform sentiment analysis (positive, neutral, negative).\n"
#     "Respond ONLY in valid JSON with keys: summary, actions, decisions, risks, translation, sentiment."
# )

# USER_TEMPLATE = (
#     "Meeting transcript:\n"
#     "{{TRANSCRIPT}}\n\n"
#     "Target language: {{LANG}}"
# )

# class LLMNLP:
#     def __init__(self):
#         self.gemini_key = os.getenv("GEMINI_API_KEY")
#         self.openai_key = os.getenv("OPENAI_API_KEY")

#         if self.gemini_key:
#             genai.configure(api_key=self.gemini_key)
#             self.provider = "gemini"
#             self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
#         elif self.openai_key and _has_openai:
#             self.provider = "openai"
#             self.openai_client = OpenAI(api_key=self.openai_key)
#             self.openai_model = "gpt-4o-mini"
#         else:
#             raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

#     def analyze(self, transcript: str, target_lang: str) -> Dict:
#         user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", transcript[:25000]).replace("{{LANG}}", target_lang)

#         if self.provider == "gemini":
#             resp = self.gemini_model.generate_content([
#                 {"role": "user", "parts": [SYSTEM_PROMPT]},
#                 {"role": "user", "parts": [user]},
#             ])
#             text = resp.text
#         else:  # OpenAI
#             msg = [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user},
#             ]
#             resp = self.openai_client.chat.completions.create(
#                 model=self.openai_model,
#                 messages=msg,
#                 temperature=0.2,
#             )
#             text = resp.choices[0].message.content

#         # Parse JSON safely
#         try:
#             data = json.loads(text)
#         except json.JSONDecodeError:
#             start = text.find('{')
#             end = text.rfind('}')
#             data = json.loads(text[start:end+1])
#         return data



# import os
# import json
# from typing import Dict
# import google.generativeai as genai

# # OpenAI (optional fallback)
# try:
#     from openai import OpenAI
#     _has_openai = True
# except Exception:
#     _has_openai = False


# # SYSTEM_PROMPT = (
# #     "You are an AI meeting summarizer.\n"
# #     "1) Summarize the meeting into 3–5 bullet points.\n"
# #     "2) Extract key action items.\n"
# #     "3) List any decisions made.\n"
# #     "4) Identify risks.\n"
# #     "5) Provide a fluent translation of the summary in the target language.\n"
# #     "6) Perform sentiment analysis (positive, neutral, negative).\n"
# #     "Respond ONLY in valid JSON with keys: summary, actions, decisions, risks, translation, sentiment."
# # )
# SYSTEM_PROMPT = """
# You are an advanced AI meeting summarizer.

# Follow these rules:

# 1. Create a detailed structured meeting summary.
#    - If transcript is longer than 5 minutes, produce at least 10 bullet points.
#    - Include major theme sections: Overview, Discussion Topics, Key Points.
#    - Add timeline cues if possible (e.g., "At 02:15 discussion about budget”).

# 2. Extract clear actionable items with ownership (who must do what, and by when).

# 3. List any decisions made in the meeting.

# 4. Identify risks, blockers, or unresolved questions.

# 5. Provide a translated version of the final summary in the target language.

# 6. Perform sentiment analysis with labels: positive, neutral, negative.

# 7. You MUST NOT invent facts. Only summarize what exists in the transcript.

# Respond ONLY in valid JSON with keys:
# summary, actions, decisions, risks, translation, sentiment.
# """




# USER_TEMPLATE = (
#     "Meeting transcript:\n"
#     "{{TRANSCRIPT}}\n\n"
#     "Target language: {{LANG}}"
# )


# class LLMNLP:

#     def __init__(self):
#         self.gemini_key = os.getenv("GEMINI_API_KEY")
#         self.openai_key = os.getenv("OPENAI_API_KEY")

#         if self.gemini_key:
#             genai.configure(api_key=self.gemini_key)
#             self.provider = "gemini"
#             # Updated working Gemini model name
#             self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
#         elif self.openai_key and _has_openai:
#             self.provider = "openai"
#             self.openai_client = OpenAI(api_key=self.openai_key)
#             self.openai_model = "gpt-4o-mini"
#         else:
#             raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

#     # ----------------------------------------------------------
#     # 🔹 Helper: Break into safe-size paragraph chunks
#     # ----------------------------------------------------------
#     def _chunk_paragraphs(self, text: str, max_chars: int = 3000) -> list[str]:
#         paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
#         chunks = []
#         current = ""

#         for p in paragraphs:
#             if len(current) + len(p) + 1 <= max_chars:
#                 current = (current + "\n" + p) if current else p
#             else:
#                 chunks.append(current)
#                 current = p

#         if current:
#             chunks.append(current)

#         if not chunks:  # failsafe
#             for i in range(0, len(text), max_chars):
#                 chunks.append(text[i:i+max_chars])

#         return chunks

#     # ----------------------------------------------------------
#     # 🔹 Helper: Summarize each chunk independently
#     # ----------------------------------------------------------
#     def _summarize_chunk(self, chunk_text: str) -> str:
#         prompt = (
#             "You are an AI meeting note-taker.\n"
#             "Summarize the following meeting segment into 3–6 concise bullet points.\n"
#             "Return ONLY the bullet points, no extra explanation.\n\n"
#             f"{chunk_text}"
#         )

#         if self.provider == "gemini":
#             resp = self.gemini_model.generate_content(prompt)
#             return (resp.text or "").strip()

#         else:  # OpenAI
#             msg = [
#                 {"role": "system", "content": "You are a helpful meeting note-taker."},
#                 {"role": "user", "content": prompt},
#             ]
#             resp = self.openai_client.chat.completions.create(
#                 model=self.openai_model,
#                 messages=msg,
#                 temperature=0.2,
#             )
#             return resp.choices[0].message.content.strip()

#     # ----------------------------------------------------------
#     # 🔹 Main hierarchical summarization entrypoint
#     # ----------------------------------------------------------
#     def analyze(self, transcript: str, target_lang: str) -> Dict:

#         # ---------- A) SHORT CASE (under ~4000 chars): use original flow ----------
#         if len(transcript) <= 4000:
#             user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", transcript).replace("{{LANG}}", target_lang)
#             return self._run_final_summary(user)

#         # ---------- B) LONG CASE: hierarchical summarization ----------
#         chunks = self._chunk_paragraphs(transcript)

#         partial_summaries = []
#         for idx, c in enumerate(chunks):
#             summary = self._summarize_chunk(c)
#             partial_summaries.append(f"Part {idx+1}:\n{summary}")

#         combined_summary = "\n\n".join(partial_summaries)

#         # Limit final LLM request size (failsafe)
#         final_input = combined_summary[:25000]

#         user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", final_input).replace("{{LANG}}", target_lang)

#         return self._run_final_summary(user)

#     # ----------------------------------------------------------
#     # 🔹 Final Stage: Convert summary → structured JSON
#     # ----------------------------------------------------------
#     def _run_final_summary(self, user_prompt: str) -> Dict:

#         if self.provider == "gemini":
#             resp = self.gemini_model.generate_content([
#                 {"role": "user", "parts": [SYSTEM_PROMPT]},
#                 {"role": "user", "parts": [user_prompt]},
#             ])
#             text = resp.text

#         else:  # OpenAI fallback
#             msg = [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_prompt},
#             ]
#             resp = self.openai_client.chat.completions.create(
#                 model=self.openai_model,
#                 messages=msg,
#                 temperature=0.2,
#             )
#             text = resp.choices[0].message.content

#         return self._parse_json(text)

#     # ----------------------------------------------------------
#     # 🔹 JSON parsing helper (safe)
#     # ----------------------------------------------------------
#     def _parse_json(self, text: str) -> Dict:
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             start = text.find("{")
#             end = text.rfind("}")
#             if start != -1 and end != -1:
#                 return json.loads(text[start:end+1])

#         # Fallback skeleton if parsing fails
#         return {
#             "summary": [],
#             "actions": [],
#             "decisions": [],
#             "risks": [],
#             "translation": "",
#             "sentiment": "neutral",
#         }




# # import os
# # import json
# # from typing import Dict
# # import google.generativeai as genai

# # # GA Optimizer Import
# # from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # # OpenAI (optional fallback)
# # try:
# #     from openai import OpenAI
# #     _has_openai = True
# # except Exception:
# #     _has_openai = False


# # SYSTEM_PROMPT = """
# # You are an advanced AI meeting summarizer.

# # Follow these rules:

# # 1. Create a detailed structured meeting summary.
# #    - If transcript is longer than 5 minutes, produce at least 10 bullet points.
# #    - Include major theme sections: Overview, Discussion Topics, Key Points.
# #    - Add timeline cues if possible (e.g., "At 02:15 discussion about budget”).

# # 2. Extract clear actionable items with ownership (who must do what, and by when).

# # 3. List any decisions made in the meeting.

# # 4. Identify risks, blockers, or unresolved questions.

# # 5. Provide a translated version of the final summary in the target language.

# # 6. Perform sentiment analysis with labels: positive, neutral, negative.

# # 7. You MUST NOT invent facts. Only summarize what exists in the transcript.

# # Respond ONLY in valid JSON with keys:
# # summary, actions, decisions, risks, translation, sentiment.
# # """


# # USER_TEMPLATE = (
# #     "Meeting transcript:\n"
# #     "{{TRANSCRIPT}}\n\n"
# #     "Target language: {{LANG}}"
# # )


# # class LLMNLP:

# #     def __init__(self):
# #         self.gemini_key = os.getenv("GEMINI_API_KEY")
# #         self.openai_key = os.getenv("OPENAI_API_KEY")

# #         if self.gemini_key:
# #             genai.configure(api_key=self.gemini_key)
# #             self.provider = "gemini"
# #             self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# #         elif self.openai_key and _has_openai:
# #             self.provider = "openai"
# #             self.openai_client = OpenAI(api_key=self.openai_key)
# #             self.openai_model = "gpt-4o-mini"

# #         else:
# #             raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

# #         # Initialize Genetic Algorithm optimizer
# #         self.ga = GeneticOptimizer(population_size=5, generations=5)


# #     # ----------------------------------------------------------
# #     # 🔹 Helper: Break transcript into chunks
# #     # ----------------------------------------------------------
# #     def _chunk_paragraphs(self, text: str, max_chars: int = 3000) -> list[str]:
# #         paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
# #         chunks = []
# #         current = ""

# #         for p in paragraphs:
# #             if len(current) + len(p) + 1 <= max_chars:
# #                 current = (current + "\n" + p) if current else p
# #             else:
# #                 chunks.append(current)
# #                 current = p

# #         if current:
# #             chunks.append(current)

# #         if not chunks:
# #             for i in range(0, len(text), max_chars):
# #                 chunks.append(text[i:i+max_chars])

# #         return chunks


# #     # ----------------------------------------------------------
# #     # 🔹 Helper: Summarize chunk independently
# #     # ----------------------------------------------------------
# #     def _summarize_chunk(self, chunk_text: str) -> str:
# #         prompt = (
# #             "You are an AI meeting note-taker.\n"
# #             "Summarize the following meeting segment into 3–6 concise bullet points.\n"
# #             "Return ONLY the bullet points, no extra explanation.\n\n"
# #             f"{chunk_text}"
# #         )

# #         if self.provider == "gemini":
# #             resp = self.gemini_model.generate_content(prompt)
# #             return (resp.text or "").strip()

# #         else:
# #             msg = [
# #                 {"role": "system", "content": "You are a helpful meeting note-taker."},
# #                 {"role": "user", "content": prompt},
# #             ]
# #             resp = self.openai_client.chat.completions.create(
# #                 model=self.openai_model,
# #                 messages=msg,
# #                 temperature=0.2,
# #             )
# #             return resp.choices[0].message.content.strip()


# #     # ----------------------------------------------------------
# #     # 🔹 Hierarchical summary logic
# #     # ----------------------------------------------------------
# #     def analyze(self, transcript: str, target_lang: str) -> Dict:

# #         # SHORT CASE - direct summary
# #         if len(transcript) <= 4000:
# #             user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", transcript).replace("{{LANG}}", target_lang)
# #             return self._run_final_summary(user)

# #         # LONG CASE - hierarchical summarization
# #         chunks = self._chunk_paragraphs(transcript)
# #         partial_summaries = []

# #         for idx, c in enumerate(chunks):
# #             summary = self._summarize_chunk(c)
# #             partial_summaries.append(f"Part {idx+1}:\n{summary}")

# #         combined_summary = "\n\n".join(partial_summaries)
# #         final_input = combined_summary[:25000]

# #         user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", final_input).replace("{{LANG}}", target_lang)

# #         return self._run_final_summary(user)


# #     # ----------------------------------------------------------
# #     # 🔥 NEW: Summarize using GA Optimization
# #     # ----------------------------------------------------------
# #     def analyze_with_ga(self, transcript: str, target_lang: str, num_candidates: int = 3) -> Dict:
# #         """
# #         Generate multiple candidate summaries, then apply
# #         Genetic Algorithm to select the best one.
# #         """
# #         candidates = []
# #         results = []

# #         for _ in range(num_candidates):
# #             result = self.analyze(transcript, target_lang)
# #             results.append(result)

# #             summary_text = " ".join(result.get("summary", []))
# #             candidates.append(summary_text)

# #         # Run GA to find best summary candidate
# #         best_summary, score = self.ga.evolve(candidates, transcript)

# #         # Attach metadata
# #         best_result = results[0]
# #         best_result["ga_fitness_score"] = score
# #         best_result["candidate_count"] = len(candidates)
# #         best_result["optimized_summary"] = best_summary

# #         return best_result


# #     # ----------------------------------------------------------
# #     # 🔹 Final LLM → JSON parsing
# #     # ----------------------------------------------------------
# #     def _run_final_summary(self, user_prompt: str) -> Dict:

# #         if self.provider == "gemini":
# #             resp = self.gemini_model.generate_content([
# #                 {"role": "user", "parts": [SYSTEM_PROMPT]},
# #                 {"role": "user", "parts": [user_prompt]},
# #             ])
# #             text = resp.text

# #         else:
# #             msg = [
# #                 {"role": "system", "content": SYSTEM_PROMPT},
# #                 {"role": "user", "content": user_prompt},
# #             ]
# #             resp = self.openai_client.chat.completions.create(
# #                 model=self.openai_model,
# #                 messages=msg,
# #                 temperature=0.2,
# #             )
# #             text = resp.choices[0].message.content

# #         return self._parse_json(text)


# #     # ----------------------------------------------------------
# #     # 🔹 JSON parser (safe load)
# #     # ----------------------------------------------------------
# #     def _parse_json(self, text: str) -> Dict:
# #         try:
# #             return json.loads(text)
# #         except json.JSONDecodeError:
# #             start = text.find("{")
# #             end = text.rfind("}")
# #             if start != -1 and end != -1:
# #                 return json.loads(text[start:end+1])

# #         return {
# #             "summary": [],
# #             "actions": [],
# #             "decisions": [],
# #             "risks": [],
# #             "translation": "",
# #             "sentiment": "neutral",
# #         }


# import os
# import json
# from typing import Dict
# import google.generativeai as genai

# # GA Optimizer Import
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # OpenAI (optional fallback)
# try:
#     from openai import OpenAI
#     _has_openai = True
# except Exception:
#     _has_openai = False


# SYSTEM_PROMPT = """
# You are an advanced AI meeting summarizer.

# Follow these rules:

# 1. Create a detailed structured meeting summary.
#    - If transcript is longer than 5 minutes, produce at least 10 bullet points.
#    - Include major theme sections: Overview, Discussion Topics, Key Points.
#    - Add timeline cues if possible (e.g., "At 02:15 discussion about budget”).

# 2. Extract clear actionable items with ownership (who must do what, and by when).

# 3. List any decisions made in the meeting.

# 4. Identify risks, blockers, or unresolved questions.

# 5. Provide a translated version of the final summary in the target language.

# 6. Perform sentiment analysis with labels: positive, neutral, negative.

# 7. You MUST NOT invent facts. Only summarize what exists in the transcript.

# Respond ONLY in valid JSON with keys:
# summary, actions, decisions, risks, translation, sentiment.
# """


# USER_TEMPLATE = (
#     "Meeting transcript:\n"
#     "{{TRANSCRIPT}}\n\n"
#     "Target language: {{LANG}}"
# )


# class LLMNLP:

#     def __init__(self):
#         self.gemini_key = os.getenv("GEMINI_API_KEY")
#         self.openai_key = os.getenv("OPENAI_API_KEY")

#         # Model selection
#         if self.gemini_key:
#             genai.configure(api_key=self.gemini_key)
#             self.provider = "gemini"
#             self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")

#         elif self.openai_key and _has_openai:
#             self.provider = "openai"
#             self.openai_client = OpenAI(api_key=self.openai_key)
#             self.openai_model = "gpt-4o-mini"

#         else:
#             raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

#         # Initialize Genetic Algorithm optimizer
#         self.ga = GeneticOptimizer(population_size=5, generations=5)


#     # ----------------------------------------------------------
#     # 🔹 Helper: Break transcript into chunks
#     # ----------------------------------------------------------
#     def _chunk_paragraphs(self, text: str, max_chars: int = 3000) -> list[str]:
#         paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
#         chunks = []
#         current = ""

#         for p in paragraphs:
#             if len(current) + len(p) <= max_chars:
#                 current = (current + "\n" + p) if current else p
#             else:
#                 chunks.append(current)
#                 current = p

#         if current:
#             chunks.append(current)

#         if not chunks:
#             for i in range(0, len(text), max_chars):
#                 chunks.append(text[i:i+max_chars])

#         return chunks


#     # ----------------------------------------------------------
#     # 🔹 Helper: Summarize each chunk
#     # ----------------------------------------------------------
#     def _summarize_chunk(self, chunk_text: str) -> str:
#         prompt = (
#             "You are an AI assistant.\n"
#             "Summarize the following into 3–6 short bullet points:\n\n"
#             f"{chunk_text}"
#         )

#         if self.provider == "gemini":
#             return (self.gemini_model.generate_content(prompt).text or "").strip()

#         msg = [
#             {"role": "system", "content": "Assistant summarizer."},
#             {"role": "user", "content": prompt},
#         ]
#         resp = self.openai_client.chat.completions.create(
#             model=self.openai_model,
#             messages=msg,
#             temperature=0.2,
#         )
#         return resp.choices[0].message.content.strip()


#     # ----------------------------------------------------------
#     # 🔥 Full Hierarchical + GA Flow
#     # ----------------------------------------------------------
#     def analyze_with_ga(self, transcript: str, target_lang: str, num_candidates: int = 3) -> Dict:
        
#         # STEP 1: Chunk transcript and generate base summary once
#         chunks = self._chunk_paragraphs(transcript)
#         base_summary_sections = [self._summarize_chunk(c) for c in chunks]
#         combined_summary = "\n".join(base_summary_sections)
        
#         # STEP 2: Generate multiple candidate summaries (reduced cost)
#         candidates = [combined_summary]  
#         results = []

#         for _ in range(num_candidates):
#             result = self.analyze(transcript, target_lang)
#             results.append(result)
#             summary_text = " ".join(result.get("summary", []))
#             candidates.append(summary_text)

#         # STEP 3: Genetic Algorithm optimization
#         best_summary, score = self.ga.evolve(list(set(candidates)), transcript)

#         # STEP 4: Attach best summary to final JSON
#         best_final = results[0]
#         best_final["optimized_summary"] = best_summary
#         best_final["ga_fitness_score"] = round(score * 100, 2)

#         # Replace original summary
#         best_final["summary"] = best_summary.split("\n")

#         return best_final


#     # ----------------------------------------------------------
#     # 🔹 Run raw LLM summary
#     # ----------------------------------------------------------
#     def analyze(self, transcript: str, target_lang: str) -> Dict:

#         if len(transcript) <= 4000:
#             user_prompt = USER_TEMPLATE.replace("{{TRANSCRIPT}}", transcript).replace("{{LANG}}", target_lang)
#         else:
#             chunks = self._chunk_paragraphs(transcript)
#             summarized = "\n".join([self._summarize_chunk(c) for c in chunks])
#             user_prompt = USER_TEMPLATE.replace("{{TRANSCRIPT}}", summarized).replace("{{LANG}}", target_lang)

#         return self._run_final_summary(user_prompt)


#     # ----------------------------------------------------------
#     # JSON Parsing
#     # ----------------------------------------------------------
#     def _run_final_summary(self, user_prompt: str) -> Dict:

#         if self.provider == "gemini":
#             text = self.gemini_model.generate_content([
#                 {"role": "user", "parts": [SYSTEM_PROMPT]},
#                 {"role": "user", "parts": [user_prompt]},
#             ]).text
#         else:
#             resp = self.openai_client.chat.completions.create(
#                 model=self.openai_model,
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=0.2,
#             )
#             text = resp.choices[0].message.content

#         return self._parse_json(text)


#     def _parse_json(self, text: str) -> Dict:
#         try:
#             return json.loads(text)
#         except:
#             start, end = text.find("{"), text.rfind("}")
#             if start != -1 and end != -1:
#                 return json.loads(text[start:end+1])

#         return {
#             "summary": [],
#             "actions": [],
#             "decisions": [],
#             "risks": [],
#             "translation": "",
#             "sentiment": "neutral"
#         }


import os
import json
from typing import Dict, List
import google.generativeai as genai

from bio_algorithms.genetic_optimizer import GeneticOptimizer  # ⭐ ADDED

# OpenAI (optional fallback)
try:
    from openai import OpenAI
    _has_openai = True
except Exception:
    _has_openai = False


SYSTEM_PROMPT = """
You are an advanced AI meeting summarizer.

Follow these rules:

1. Create a detailed structured meeting summary.
   - If transcript is longer than 5 minutes, produce at least 10 bullet points.
   - Include major theme sections: Overview, Discussion Topics, Key Points.
   - Add timeline cues if possible (e.g., "At 02:15 discussion about budget”).

2. Extract clear actionable items with ownership (who must do what, and by when).

3. List any decisions made in the meeting.

4. Identify risks, blockers, or unresolved questions.

5. Provide a translated version of the final summary in the target language.

6. Perform sentiment analysis with labels: positive, neutral, negative.

7. You MUST NOT invent facts. Only summarize what exists in the transcript.

Respond ONLY in valid JSON with keys:
summary, actions, decisions, risks, translation, sentiment.
"""


USER_TEMPLATE = (
    "Meeting transcript:\n"
    "{{TRANSCRIPT}}\n\n"
    "Target language: {{LANG}}"
)


class LLMNLP:

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # LLM Source selection
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.provider = "gemini"
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        elif self.openai_key and _has_openai:
            self.provider = "openai"
            self.openai_client = OpenAI(api_key=self.openai_key)
            self.openai_model = "gpt-4o-mini"
        else:
            raise RuntimeError("No LLM provider configured.")

        # ⭐ Initialize Genetic Algorithm
        self.ga = GeneticOptimizer(population_size=5, generations=5, mutation_rate=0.2)


    # ---------------------------------------------
    # Text Chunking helper
    # ---------------------------------------------
    def _chunk_paragraphs(self, text: str, max_chars: int = 3000) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, current = [], ""

        for p in paragraphs:
            if len(current) + len(p) + 1 <= max_chars:
                current = (current + "\n" + p) if current else p
            else:
                chunks.append(current)
                current = p

        if current:
            chunks.append(current)

        if not chunks:
            return [text]

        return chunks


    # ---------------------------------------------
    # Summarize each chunk
    # ---------------------------------------------
    def _summarize_chunk(self, chunk_text: str) -> str:
        prompt = (
            "You are an AI assistant.\n"
            "Summarize this part of the meeting into 3–6 bullet points:\n\n"
            f"{chunk_text}"
        )

        if self.provider == "gemini":
            resp = self.gemini_model.generate_content(prompt)
            return (resp.text or "").strip()

        else:
            resp = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()


    # ---------------------------------------------
    # Main LLM Summarization (no GA)
    # ---------------------------------------------
    def analyze(self, transcript: str, target_lang: str) -> Dict:

        if len(transcript) <= 4000:
            user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", transcript).replace("{{LANG}}", target_lang)
            return self._run_final_summary(user)

        chunks = self._chunk_paragraphs(transcript)
        summaries = [self._summarize_chunk(c) for c in chunks]

        combined = "\n\n".join(summaries)
        user = USER_TEMPLATE.replace("{{TRANSCRIPT}}", combined[:25000]).replace("{{LANG}}", target_lang)

        return self._run_final_summary(user)


    # ---------------------------------------------
    # ⭐ GA Optimization Version
    # ---------------------------------------------
    def analyze_with_ga(self, transcript: str, target_lang: str, candidate_count: int = 3) -> Dict:

        # Generate multiple candidate summaries
        candidate_results: List[Dict] = []
        candidates = []

        for _ in range(candidate_count):
            result = self.analyze(transcript, target_lang)
            candidate_results.append(result)
            candidates.append(" ".join(result.get("summary", [])))

        # Run GA optimization
        best_summary, score = self.ga.evolve(candidates, transcript)

        # Attach GA metadata
        best_result = candidate_results[0]
        best_result["optimized_summary"] = best_summary
        best_result["ga_fitness_score"] = round(score * 100, 2)  # Convert to %
        best_result["candidate_count"] = candidate_count

        return best_result


    # ---------------------------------------------
    # LLM JSON handling
    # ---------------------------------------------
    def _run_final_summary(self, user_prompt: str) -> Dict:

        if self.provider == "gemini":
            resp = self.gemini_model.generate_content([
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                {"role": "user", "parts": [user_prompt]},
            ])
            text = resp.text

        else:
            resp = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content

        return self._parse_json(text)


    def _parse_json(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])

        return {
            "summary": [],
            "actions": [],
            "decisions": [],
            "risks": [],
            "translation": "",
            "sentiment": "neutral",
        }

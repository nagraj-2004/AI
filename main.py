# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()

# # Config
# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # 🔹 Helper: split audio into safe chunks
# def chunk_audio_file(audio_path, chunk_length_ms=60_000):
#     """Split audio into fixed chunks (default 60s)."""
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks


# # 🔹 Helper: safe TTS with chunking
# def chunked_tts(text, lang="en", max_chars=2500, out_path="outputs/summary_audio.mp3"):
#     """Split long text into safe chunks for gTTS and merge into one MP3."""
#     os.makedirs("outputs", exist_ok=True)
#     parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
#     audio_segments = []

#     for idx, part in enumerate(parts):
#         tts = gTTS(part, lang=lang)
#         temp_path = f"outputs/tmp_tts_{idx}.mp3"
#         tts.save(temp_path)
#         audio_segments.append(AudioSegment.from_mp3(temp_path))

#     combined = sum(audio_segments)
#     combined.export(out_path, format="mp3")
#     return out_path


# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     status_msgs = []

#     def update_status(msg):
#         status_msgs.append(msg)
#         return "\n".join(status_msgs)

#     # 1. Check input
#     if not audio_path:
#         return "", "", "No audio uploaded", "", None, "(No auto insight)", "(No chatbot query)", update_status("❌ No audio uploaded")

#     # 2. Language detection
#     update_status("⏳ Detecting language...")
#     source_lang = transcriber.detect_language(audio_path)
#     update_status(f"🗣️ Detected language: {source_lang}")

#     # 3. Transcription with chunking
#     update_status("🎙️ Transcribing audio (chunked)...")
#     try:
#         chunks = chunk_audio_file(audio_path, chunk_length_ms=60_000)
#         transcripts = []
#         for ch in chunks:
#             result = transcriber.transcribe(ch, language=source_lang)
#             transcripts.append(result.get("text", ""))
#         transcript = " ".join(transcripts).strip()
#     except Exception as e:
#         return source_lang, "", "Error in transcription", "", None, "(No auto insight)", "(No chatbot query)", update_status(f"❌ Transcription error: {e}")

#     if not transcript:
#         return source_lang, "", "Empty transcript", "", None, "(No auto insight)", "(No chatbot query)", update_status("❌ Transcript is empty")

#     # 4. AI insights
#     update_status("🤖 Generating AI insights...")
#     gemini_auto_response = get_gemini_response(transcript)
#     gemini_chat_response = get_gemini_response(custom_query) if custom_query.strip() else "(No custom query provided)"

#     # 5. NLP analysis
#     update_status("🔍 Analyzing transcript with NLP...")
#     analysis = nlp.analyze(transcript, target_lang)
#     summary_bullets = analysis.get("summary", [])
#     action_items = analysis.get("actions", [])
#     decisions = analysis.get("decisions", [])
#     risks = analysis.get("risks", [])
#     sentiment = analysis.get("sentiment", "neutral")

#     # ✅ FIX: no more transcript[:2000], use full transcript
#     translated = (
#         analysis.get("translation")
#         or (" ".join(summary_bullets) if summary_bullets else transcript)
#     )

#     if isinstance(translated, dict):
#         translated = translated.get(target_lang) or translated.get("en") or str(translated)

#     # 6. TTS with chunking
#     update_status("🔊 Generating TTS summary audio...")
#     os.makedirs("outputs", exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     tts_path = os.path.join("outputs", f"summary_audio_{timestamp}.mp3")

#     try:
#         tts_path = chunked_tts(translated, lang=target_lang if target_lang else "en", out_path=tts_path)
#     except Exception as e:
#         update_status(f"⚠️ TTS error: {e}")
#         tts_path = None

#     # 7. Build final summary text
#     text_block = ["**Meeting Summary**"] + [f"• {b}" for b in summary_bullets]
#     text_block += ["\n**Action Items**"] + ([f"- {a}" for a in action_items] if action_items else ["- (none)"])
#     text_block += ["\n**Decisions**"] + ([f"- {d}" for d in decisions] if decisions else ["- (none)"])
#     text_block += ["\n**Risks**"] + ([f"- {r}" for r in risks] if risks else ["- (none)"])
#     text_block.append(f"\n**Sentiment:** {sentiment}")
#     text_block.append("\n**Gemini Auto Insight:**")
#     text_block.append(gemini_auto_response)
#     share_text = "\n".join(text_block)

#     # 8. Notifications
#     update_status("📤 Sending Slack and Email notifications...")

#     # Slack
#     try:
#         send_slack_message(share_text)
#         update_status("✅ Slack message sent.")
#     except Exception as e:
#         update_status(f"⚠️ Slack error: {e}")

#     # Email
#     try:
#         final_recipients = RECIPIENTS.copy() if RECIPIENTS else []
#         if extra_emails:
#             emails = [e.strip() for e in extra_emails.split(",") if "@" in e]
#             final_recipients.extend(emails)
#         if final_recipients:
#             send_email(subject="Meeting Summary", body=share_text, recipients=final_recipients)
#             update_status(f"✅ Email sent to: {', '.join(final_recipients)}")
#         else:
#             update_status("⚠️ No recipients found, email not sent.")
#     except Exception as e:
#         update_status(f"⚠️ Email error: {e}")

#     update_status("🎉 Processing complete!")

#     return (
#         source_lang,
#         transcript,
#         share_text,
#         translated,
#         tts_path,
#         gemini_auto_response,
#         gemini_chat_response,
#         "\n".join(status_msgs),
#     )

# # Gradio UI
# with gr.Blocks(css="""
#     footer {visibility: hidden}
#     #logo img {max-height: 120px; object-fit: contain;}  /* ✅ show full logo */
#     .gr-button {background: #2563eb !important; color: white !important; font-weight: bold;}
# """) as ui:

#     # 🔹 Header with Logo + Title
#     with gr.Row():
#         with gr.Column(scale=1):
#             if os.path.exists("assets/logo.png"):
#                 gr.Image(
#                     value="assets/logo.png",
#                     elem_id="logo",
#                     show_label=False,
#                     type="filepath",
#                     interactive=False   # ✅ prevents cropping UI
#                 )
#         with gr.Column(scale=4):
#             gr.Markdown("## 🚀 AI Meeting Summarizer\n Dashboard with Insights, Audio & Export")

#     with gr.Row():
#         # 📥 Input Panel
#         with gr.Column(scale=1):
#             audio_input = gr.Audio(label="🎙️ Upload Meeting Audio", type="filepath")
#             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#             custom_query = gr.Textbox(label="💡 Ask AI Insight", placeholder="Type your question here...")
#             extra_emails = gr.Textbox(label="📧 Extra Emails", placeholder="Enter emails separated by commas")
#             submit_btn = gr.Button("🔎 Process Audio", variant="primary")

#         # 📤 Output Panel
#         with gr.Column(scale=2):
#             status_display = gr.Markdown("⏳ Waiting for input...")

#             with gr.Tab("📄 Transcript"):
#                 output_lang = gr.Textbox(label="Detected Language", interactive=False)
#                 output_transcript = gr.Textbox(
#                     label="Transcript",
#                     lines=5,    
#                     max_lines=30,  # expands dynamically
#                     interactive=False,
#                     show_copy_button=True
#                 )
#                 download_transcript = gr.File(label="⬇️ Download Transcript")

#             with gr.Tab("📝 Summary"):
#                 output_summary = gr.Markdown()
#                 download_summary = gr.File(label="⬇️ Download Summary")

#             with gr.Tab("🌐 Translation"):
#                 output_translated = gr.Textbox(
#                     label="Translated Summary",
#                     lines=10,
#                     interactive=False,
#                     show_copy_button=True
#                 )

#             with gr.Tab("🔊 Audio Summary"):
#                 output_tts = gr.Audio(label="Download Summary Audio", type="filepath")

#             with gr.Tab("🤖 AI Insights"):
#                 output_auto = gr.Textbox(label="Gemini Auto Insight", lines=5, interactive=False)
#                 output_chat = gr.Textbox(label="Gemini Chatbot Response", lines=5, interactive=False)

#     # 🔗 Helper to save files
#     def save_files(lang, transcript, summary):
#         """Save transcript & summary as downloadable files."""
#         os.makedirs("outputs", exist_ok=True)
#         ts_path, sm_path = None, None
#         if transcript:
#             ts_path = os.path.join("outputs", "transcript.txt")
#             with open(ts_path, "w", encoding="utf-8") as f:
#                 f.write(transcript)
#         if summary:
#             sm_path = os.path.join("outputs", "summary.txt")
#             with open(sm_path, "w", encoding="utf-8") as f:
#                 f.write(summary)
#         return ts_path, sm_path

#     # 🔗 Button actions
#     submit_btn.click(
#         fn=pipeline_with_status,
#         inputs=[audio_input, lang_input, custom_query, extra_emails],
#         outputs=[
#             output_lang,
#             output_transcript,
#             output_summary,
#             output_translated,
#             output_tts,
#             output_auto,
#             output_chat,
#             status_display,
#         ],
#     ).then(
#         fn=save_files,
#         inputs=[output_lang, output_transcript, output_summary],
#         outputs=[download_transcript, download_summary],
#     )

# if __name__ == "__main__":
#     ui.launch()







# # import os
# # from dotenv import load_dotenv
# # import gradio as gr
# # from gtts import gTTS
# # from datetime import datetime
# # from pydub import AudioSegment
# # import matplotlib.pyplot as plt   # ⭐ ADDED FOR GA
# # import numpy as np                # ⭐ ADDED FOR GA

# # # Agents & integrations
# # from agents.transcriber import TranscriberAgent
# # from agents.llm_nlp import LLMNLP
# # from agents.highlighter import HighlightAgent
# # from integrations.slack_notify import send_slack_message
# # from integrations.emailer import send_email, RECIPIENTS
# # from integrations.gemini_api import get_gemini_response

# # load_dotenv()

# # # Agents
# # transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# # nlp = LLMNLP()
# # highlighter = HighlightAgent()

# # # Config
# # LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# # DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # # ---------- AUDIO CHUNKING ----------
# # def chunk_audio_file(audio_path, chunk_length_ms=60_000):
# #     audio = AudioSegment.from_file(audio_path)
# #     return [
# #         (audio[i:i+chunk_length_ms]).export(f"{audio_path}_chunk_{i//chunk_length_ms}.wav",
# #         format="wav") or f"{audio_path}_chunk_{i//chunk_length_ms}.wav"
# #         for i in range(0, len(audio), chunk_length_ms)
# #     ]


# # # ---------- TTS SAFE ----------
# # def chunked_tts(text, lang="en", max_chars=2500, out_path="outputs/summary_audio.mp3"):
# #     os.makedirs("outputs", exist_ok=True)
# #     parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
# #     audio = sum([AudioSegment.from_mp3(save_part(part, i)) for i, part in enumerate(parts)], AudioSegment.empty())

# #     def save_part(txt, idx):
# #         temp = f"outputs/tmp_{idx}.mp3"
# #         gTTS(txt, lang=lang).save(temp)
# #         return temp

# #     audio.export(out_path, format="mp3")
# #     return out_path



# # # ---------- PIPELINE ----------
# # def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
# #     status = []

# #     def push(msg):
# #         status.append(msg)
# #         return "\n".join(status)

# #     if not audio_path:
# #         return "", "", "", "", None, "", "", "❌ No audio uploaded", None

# #     push("⏳ Detecting language...")
# #     lang = transcriber.detect_language(audio_path)

# #     push("🎙 Transcribing...")
# #     chunks = chunk_audio_file(audio_path)
# #     transcript = " ".join([transcriber.transcribe(c).get("text", "") for c in chunks]).strip()

# #     push("🤖 Running NLP + GA Optimization...")
# #     result = nlp.analyze_with_ga(transcript, target_lang)

# #     summary = "\n".join([f"• {s}" for s in result.get("summary", [])])
# #     score = result.get("ga_fitness_score", 0)

# #     # ---------- ⭐ GENERATE SCORE GRAPH ----------
# #     os.makedirs("outputs", exist_ok=True)
# #     graph_path = f"outputs/ga_score_{datetime.now().strftime('%H%M%S')}.png"

# #     fig, ax = plt.subplots(figsize=(4, 2))
# #     ax.bar(["GA Fitness Score"], [score])
# #     ax.set_ylim(0, 100)
# #     ax.set_ylabel("Score %")
# #     plt.title("Genetic Algorithm Optimization Score")
# #     plt.tight_layout()
# #     fig.savefig(graph_path)
# #     plt.close(fig)
# #     # ------------------------------------------------

# #     push("🔊 Creating Audio Summary...")
# #     tts = chunked_tts(result.get("optimized_summary", ""), target_lang)

# #     push("📤 Sending notifications...")
# #     try: send_slack_message(summary)
# #     except: pass

# #     push("🎉 Done!")

# #     return (
# #         lang,
# #         transcript,
# #         summary,
# #         result.get("translation", ""),
# #         tts,
# #         result.get("optimized_summary", ""),
# #         result.get("sentiment", ""),
# #         "\n".join(status),
# #         graph_path
# #     )


# # # ---------- UI ----------
# # with gr.Blocks(css="""
# #     footer {visibility: hidden}
# #     #logo img {max-height: 120px;}
# #     .gr-button {background:#2563eb !important;color:white !important;font-weight:bold;}
# # """) as ui:

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             if os.path.exists("assets/logo.png"):
# #                 gr.Image(value="assets/logo.png", elem_id="logo", show_label=False)
# #         with gr.Column(scale=4):
# #             gr.Markdown("## 🚀 AI Meeting Summarizer\nDashboard with GA Optimization + Insights")

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
# #             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Language")
# #             custom_query = gr.Textbox(label="💬 Ask AI", placeholder="Ask question...")
# #             extra_emails = gr.Textbox(label="📧 Extra Emails")
# #             submit_btn = gr.Button("🔎 Process", variant="primary")

# #         with gr.Column(scale=2):
# #             status_display = gr.Markdown("⏳ Waiting...")

# #             with gr.Tab("📄 Transcript"): 
# #                 output_lang = gr.Textbox(label="Detected Language", interactive=False)
# #                 output_transcript = gr.Textbox(lines=8, show_copy_button=True)

# #             with gr.Tab("📝 Summary"):
# #                 output_summary = gr.Markdown()

# #             with gr.Tab("🌐 Translation"):
# #                 output_translated = gr.Textbox(lines=10, show_copy_button=True)

# #             with gr.Tab("🔊 Audio"):
# #                 output_audio = gr.Audio(type="filepath")

# #             with gr.Tab("🤖 Optimized Summary"):
# #                 output_opt = gr.Textbox(lines=6, show_copy_button=True)

# #             with gr.Tab("📈 GA Score"):   # ⭐ NEW TAB
# #                 output_graph = gr.Image(label="GA Optimization Score Visualization")
# #                 output_sentiment = gr.Textbox(label="Sentiment Result")


# #     submit_btn.click(
# #         fn=pipeline_with_status,
# #         inputs=[audio_input, lang_input, custom_query, extra_emails],
# #         outputs=[
# #             output_lang,
# #             output_transcript,
# #             output_summary,
# #             output_translated,
# #             output_audio,
# #             output_opt,
# #             output_sentiment,
# #             status_display,
# #             output_graph    # ⭐ NEW OUTPUT
# #         ]
# #     )


# # if __name__ == "__main__":
# #     ui.launch()




# # import os
# # from dotenv import load_dotenv
# # import gradio as gr
# # from gtts import gTTS
# # from datetime import datetime
# # from pydub import AudioSegment
# # import matplotlib.pyplot as plt
# # from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # # Agents & integrations
# # from agents.transcriber import TranscriberAgent
# # from agents.llm_nlp import LLMNLP
# # from agents.highlighter import HighlightAgent
# # from integrations.slack_notify import send_slack_message
# # from integrations.emailer import send_email, RECIPIENTS
# # from integrations.gemini_api import get_gemini_response

# # load_dotenv()

# # # Agents
# # transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# # nlp = LLMNLP()
# # ga = GeneticOptimizer()
# # highlighter = HighlightAgent()

# # LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# # DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # # ----------------------------------------
# # # 🛠 FIXED: Chunk audio properly
# # # ----------------------------------------
# # def chunk_audio_file(audio_path, chunk_length_ms=60000):
# #     audio = AudioSegment.from_file(audio_path)
# #     chunk_paths = []
    
# #     for i in range(0, len(audio), chunk_length_ms):
# #         chunk = audio[i:i + chunk_length_ms]
# #         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
# #         chunk.export(path, format="wav")
# #         chunk_paths.append(path)

# #     return chunk_paths


# # # ----------------------------------------
# # # 🛠 FIXED TTS
# # # ----------------------------------------
# # def chunked_tts(text, lang="en", max_chars=2500, out_path="outputs/summary_audio.mp3"):
# #     os.makedirs("outputs", exist_ok=True)
# #     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    
# #     segments = []
# #     for idx, p in enumerate(parts):
# #         temp = f"outputs/tmp_tts_{idx}.mp3"
# #         gTTS(p, lang=lang).save(temp)
# #         segments.append(AudioSegment.from_mp3(temp))

# #     final = sum(segments)
# #     final.export(out_path, format="mp3")
# #     return out_path


# # # ----------------------------------------
# # # 🚀 Main Pipeline
# # # ----------------------------------------
# # def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
# #     logs = []

# #     def log(msg):
# #         logs.append(msg)
# #         return "\n".join(logs)

# #     if not audio_path:
# #         return "", "", "", "", None, "", "", "❌ No audio uploaded", None

# #     log("⏳ Detecting language...")
# #     lang = transcriber.detect_language(audio_path)

# #     log("🎙 Transcribing...")
# #     chunks = chunk_audio_file(audio_path)

# #     transcript = " ".join([
# #         transcriber.transcribe(c, language=lang).get("text", "")
# #         for c in chunks
# #     ]).strip()

# #     log("🤖 Running NLP with GA Optimization...")
# #     result = nlp.analyze_with_ga(transcript, target_lang)

# #     summary = "\n".join([f"• {s}" for s in result.get("summary", [])])
# #     optimized = result.get("optimized_summary", "")
# #     score = result.get("ga_fitness_score", 0)

# #     # ----------------------------------------
# #     # 📈 Create GA Score Graph
# #     # ----------------------------------------
# #     os.makedirs("outputs", exist_ok=True)
# #     graph_path = f"outputs/ga_score_{datetime.now().strftime('%H%M%S')}.png"

# #     plt.figure(figsize=(4, 2))
# #     plt.bar(["GA Score"], [score], color="blue")
# #     plt.ylim(0, 100)
# #     plt.tight_layout()
# #     plt.savefig(graph_path)
# #     plt.close()

# #     log("🔊 Creating TTS...")
# #     audio_out = chunked_tts(optimized if optimized else summary, target_lang)

# #     log("📤 Sending notifications...")
# #     try: send_slack_message(summary)
# #     except: pass

# #     log("🎉 Done!")

# #     return (
# #         lang,
# #         transcript,
# #         summary,
# #         result.get("translation", ""),
# #         audio_out,
# #         optimized,
# #         result.get("sentiment", ""),
# #         "\n".join(logs),
# #         graph_path
# #     )


# # # ----------------------------------------
# # # 🎨 UI Layout (unchanged except score tab)
# # # ----------------------------------------
# # with gr.Blocks(css="""
# #     footer {visibility: hidden}
# #     #logo img {max-height: 120px;}
# #     .gr-button {background:#2563eb !important;color:white !important;font-weight:bold;}
# # """) as ui:

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             if os.path.exists("assets/logo.png"):
# #                 gr.Image(value="assets/logo.png", elem_id="logo", show_label=False)
# #         with gr.Column(scale=4):
# #             gr.Markdown("## 🚀 AI Meeting Summarizer\nDashboard with GA Optimization + Insights")

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
# #             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Language")
# #             custom_query = gr.Textbox(label="💬 Ask AI", placeholder="Ask question...")
# #             extra_emails = gr.Textbox(label="📧 Extra Emails")
# #             submit_btn = gr.Button("🔎 Process")

# #         with gr.Column(scale=2):
# #             status_display = gr.Markdown("⏳ Waiting...")

# #             with gr.Tab("📄 Transcript"): 
# #                 output_lang = gr.Textbox(label="Detected Language")
# #                 output_transcript = gr.Textbox(lines=8, show_copy_button=True)

# #             with gr.Tab("📝 Summary"):
# #                 output_summary = gr.Markdown()

# #             with gr.Tab("🌐 Translation"):
# #                 output_translated = gr.Textbox(lines=10, show_copy_button=True)

# #             with gr.Tab("🔊 Audio"):
# #                 output_audio = gr.Audio(type="filepath")

# #             with gr.Tab("🤖 Optimized Summary"):
# #                 output_opt = gr.Textbox(lines=6, show_copy_button=True)

# #             with gr.Tab("📈 GA Score"):
# #                 output_graph = gr.Image()
# #                 output_sentiment = gr.Textbox(label="Sentiment Result")


# #     submit_btn.click(
# #         fn=pipeline_with_status,
# #         inputs=[audio_input, lang_input, custom_query, extra_emails],
# #         outputs=[
# #             output_lang,
# #             output_transcript,
# #             output_summary,
# #             output_translated,
# #             output_audio,
# #             output_opt,
# #             output_sentiment,
# #             status_display,
# #             output_graph
# #         ]
# #     )


# # if __name__ == "__main__":
# #     ui.launch()










# # import os
# # from dotenv import load_dotenv
# # import gradio as gr
# # from gtts import gTTS
# # from datetime import datetime
# # from pydub import AudioSegment
# # import matplotlib.pyplot as plt

# # # Genetic Algorithm
# # from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # # Agents & integrations
# # from agents.transcriber import TranscriberAgent
# # from agents.llm_nlp import LLMNLP
# # from agents.highlighter import HighlightAgent
# # from integrations.slack_notify import send_slack_message
# # from integrations.emailer import send_email, RECIPIENTS
# # from integrations.gemini_api import get_gemini_response

# # load_dotenv()

# # # Agents
# # transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# # nlp = LLMNLP()
# # ga = GeneticOptimizer()
# # highlighter = HighlightAgent()

# # LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# # DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # # ---------------------- AUDIO CHUNK FIX ----------------------
# # def chunk_audio_file(audio_path, chunk_length_ms=60000):
# #     audio = AudioSegment.from_file(audio_path)
# #     files = []

# #     for i in range(0, len(audio), chunk_length_ms):
# #         chunk_path = f"{audio_path}_chunk_{i//60000}.wav"
# #         audio[i:i+chunk_length_ms].export(chunk_path, format="wav")
# #         files.append(chunk_path)

# #     return files


# # # ---------------------- FIXED TTS ----------------------
# # def chunked_tts(text, lang="en", max_chars=2500, out_path="outputs/summary_audio.mp3"):
# #     text = str(text)   # 🆕 make sure text is always a string

# #     os.makedirs("outputs", exist_ok=True)
# #     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    
# #     segments = []
# #     for idx, p in enumerate(parts):
# #         temp = f"outputs/tmp_tts_{idx}.mp3"
# #         gTTS(p, lang=lang).save(temp)
# #         segments.append(AudioSegment.from_mp3(temp))

# #     final = sum(segments)
# #     final.export(out_path, format="mp3")
# #     return out_path



# # # ---------------------- MAIN PIPELINE ----------------------
# # def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
# #     logs = []

# #     def log(msg):
# #         logs.append(msg)
# #         return "\n".join(logs)

# #     if not audio_path:
# #         outputs = [""] * 10
# #         outputs[7] = "❌ No audio uploaded"  # status message
# #         return tuple(outputs)


# #     log("⏳ Detecting language...")
# #     lang = transcriber.detect_language(audio_path)

# #     log("🎙 Transcribing...")
# #     chunks = chunk_audio_file(audio_path)

# #     transcript = " ".join([
# #         transcriber.transcribe(c, language=lang).get("text", "")
# #         for c in chunks
# #     ]).strip()

# #     log("🤖 AI Analysis & GA Optimization...")
# #     result = nlp.analyze_with_ga(transcript, target_lang)

# #     summary_list = result.get("summary", [])
# #     summary = "\n".join([f"• {s}" for s in summary_list])

# #     optimized = result.get("optimized_summary", "(No optimized output)")
# #     score = result.get("ga_fitness_score", 0)
# #     sentiment = result.get("sentiment", "neutral")

# #     # ---------------------- SCORE GRAPH ----------------------
# #     os.makedirs("outputs", exist_ok=True)
# #     graph_path = f"outputs/ga_score_{datetime.now().strftime('%H%M%S')}.png"

# #     plt.figure(figsize=(4, 2))
# #     plt.bar(["GA Score"], [score])
# #     plt.ylim(0, 100)
# #     plt.title("Genetic Optimization Score")
# #     plt.tight_layout()
# #     plt.savefig(graph_path)
# #     plt.close()

# #     log("🔊 Generating Speech...")
    
# #     tts_text = result.get("translation", "") or optimized or summary
# #     audio_out = chunked_tts(tts_text, target_lang)

# #     log("📤 Sending Notifications...")
# #     try: send_slack_message(summary)
# #     except: pass

# #     log("🎉 Completed.")

# #     return (
# #         lang,
# #         transcript,
# #         summary,
# #         result.get("translation", ""),
# #         audio_out,
# #         optimized,
# #         sentiment,
# #         "\n".join(logs),
# #         graph_path,
# #         result.get("gemini_auto", "(No auto insight)"),
# #         result.get("gemini_chat", "(No chat response)")
# #     )


# # # ---------------------- UI ----------------------
# # with gr.Blocks(css="""
# #     footer {visibility: hidden}
# #     #logo img {max-height: 120px;}
# #     .gr-button {background:#2563eb !important;color:white !important;font-weight:bold;}
# # """) as ui:

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             if os.path.exists("assets/logo.png"):
# #                 gr.Image(value="assets/logo.png", elem_id="logo", show_label=False)

# #         with gr.Column(scale=4):
# #             gr.Markdown("## 🚀 AI Meeting Summarizer\nWith Bio-Inspired GA Optimization")

# #     with gr.Row():
# #         with gr.Column(scale=1):
# #             audio_input = gr.Audio(label="🎙 Upload Meeting Audio", type="filepath")
# #             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
# #             custom_query = gr.Textbox(label="💬 Ask AI", placeholder="Ask question here...")
# #             extra_emails = gr.Textbox(label="📧 Extra Emails")
# #             submit_btn = gr.Button("🔎 Process")

# #         with gr.Column(scale=2):
# #             status_display = gr.Markdown("⏳ Waiting...")

# #             with gr.Tab("📄 Transcript"):
# #                 output_lang = gr.Textbox(label="Detected Language")
# #                 output_transcript = gr.Textbox(lines=8, show_copy_button=True)

# #             with gr.Tab("📝 Summary"):
# #                 output_summary = gr.Markdown()

# #             with gr.Tab("🌐 Translation"):
# #                 output_translated = gr.Textbox(lines=10, show_copy_button=True)

# #             with gr.Tab("🔊 Audio Summary"):
# #                 output_audio = gr.Audio(type="filepath")

# #             with gr.Tab("🤖 Optimized Summary (GA)"):
# #                 output_opt = gr.Textbox(lines=6, show_copy_button=True)

# #             with gr.Tab("🧠 AI Insights"):
# #                 output_auto = gr.Textbox(label="Gemini Auto Response", lines=5)
# #                 output_chat = gr.Textbox(label="Custom Query Response", lines=5)

# #             with gr.Tab("📈 GA Score"):
# #                 output_graph = gr.Image()
# #                 output_sentiment = gr.Textbox(label="Sentiment Result")


# #     submit_btn.click(
# #         fn=pipeline_with_status,
# #         inputs=[audio_input, lang_input, custom_query, extra_emails],
# #         outputs=[
# #             output_lang, output_transcript, output_summary,
# #             output_translated, output_audio, output_opt,
# #             output_sentiment, status_display, output_graph,
# #             output_auto, output_chat
# #         ]
# #     )


# # if __name__ == "__main__":
# #     ui.launch()





# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment

# # 🧬 NEW IMPORT
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()   # 🧬 NEW

# # Config
# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # --- Helper: split audio ----
# def chunk_audio_file(audio_path, chunk_length_ms=60_000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks


# # --- Helper: TTS ---
# def chunked_tts(text, lang="en", max_chars=2500, out_path="outputs/summary_audio.mp3"):
#     os.makedirs("outputs", exist_ok=True)
#     parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
#     audio_segments = []

#     for idx, part in enumerate(parts):
#         tts = gTTS(part, lang=lang)
#         temp_path = f"outputs/tmp_tts_{idx}.mp3"
#         tts.save(temp_path)
#         audio_segments.append(AudioSegment.from_mp3(temp_path))

#     combined = sum(audio_segments)
#     combined.export(out_path, format="mp3")
#     return out_path



# # --- MAIN PIPELINE ---
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     status_msgs = []

#     def update_status(msg):
#         status_msgs.append(msg)
#         return "\n".join(status_msgs)

#     if not audio_path:
#         return tuple([""] * 10)

#     update_status("⏳ Detecting language...")
#     source_lang = transcriber.detect_language(audio_path)
#     update_status(f"🗣️ Detected language: {source_lang}")

#     update_status("🎙️ Transcribing audio (chunked)...")
#     chunks = chunk_audio_file(audio_path)
#     transcript = " ".join([transcriber.transcribe(c, language=source_lang).get("text", "") for c in chunks]).strip()

#     update_status("🤖 Analyzing transcript with NLP + GA...")
#     analysis = nlp.analyze(transcript, target_lang)

#     # ---- GA candidate pool ----
#     candidates = [
#         " ".join(analysis.get("summary", [])),
#         " ".join(analysis.get("actions", [])),
#         transcript[:500],  # transcript snippet baseline
#     ]

#     optimized_summary, ga_score = ga.evolve(candidates, transcript)

#     summary_bullets = analysis.get("summary", [])
#     sentiment = analysis.get("sentiment", "neutral")
#     translated = analysis.get("translation", "")

#     update_status("🔊 Generating audio summary...")
#     tts_path = chunked_tts(optimized_summary, target_lang)

#     update_status("🎉 Completed Successfully!")

#     return (
#         source_lang,
#         transcript,
#         share_text,
#         translated,
#         tts_path,
#         gemini_auto_response,
#         gemini_chat_response,
#         "\n".join(status_msgs),
#         "",          # placeholder for GA optimized summary if not yet added
#         ""           # placeholder for GA score graph
#     )





# # -------- UI --------
# with gr.Blocks(css='''
#     footer {visibility: hidden}
#     #logo img {max-height: 120px;}
#     .gr-button {background:#2563eb !important;color:white !important;font-weight:bold;}
# ''') as ui:

#     with gr.Row():
#         with gr.Column(scale=1):
#             if os.path.exists("assets/logo.png"):
#                 gr.Image(value="assets/logo.png", elem_id="logo", show_label=False)
#         with gr.Column(scale=4):
#             gr.Markdown("## 🚀 AI Meeting Summarizer\n Dashboard with Insights, Audio & Export")

#     with gr.Row():
#         with gr.Column(scale=1):
#             audio_input = gr.Audio(label="🎙 Upload Meeting Audio", type="filepath")
#             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#             custom_query = gr.Textbox(label="💬 Ask AI Insight")
#             extra_emails = gr.Textbox(label="📧 Extra Emails")
#             submit_btn = gr.Button("🔎 Process Audio")

#         with gr.Column(scale=2):
#             status_display = gr.Markdown("⏳ Waiting...")

#             with gr.Tab("📄 Transcript"):
#                 output_lang = gr.Textbox(label="Detected Language")
#                 output_transcript = gr.Textbox(lines=8, show_copy_button=True)

#             with gr.Tab("📝 Summary"):
#                 output_summary = gr.Markdown()

#             with gr.Tab("🌐 Translation"):
#                 output_translated = gr.Textbox(lines=10, show_copy_button=True)

#             with gr.Tab("🔊 Audio Summary"):
#                 output_tts = gr.Audio(type="filepath")

#             with gr.Tab("🤖 Optimized Summary (GA)"):
#                 output_ga = gr.Textbox(label="GA Best Summary", lines=6, show_copy_button=True)

#             with gr.Tab("📈 GA Score"):
#                 output_score = gr.Textbox(label="Fitness Score (%)", interactive=False)

#             with gr.Tab("😊 Sentiment"):
#                 output_sentiment = gr.Textbox(label="Tone")

#     submit_btn.click(
#         fn=pipeline_with_status,
#         inputs=[audio_input, lang_input, custom_query, extra_emails],
#         outputs=[
#             output_lang, output_transcript, output_summary, output_translated,
#             output_tts, output_ga, output_score, output_sentiment, status_display, None
#         ]
#     )


# if __name__ == "__main__":
#     ui.launch()




# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment
# import traceback

# # 🧬 Genetic Algorithm
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()

# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # -------- AUDIO CHUNKING --------
# def chunk_audio_file(audio_path, chunk_length_ms=60000):
#     audio = AudioSegment.from_file(audio_path)
#     paths = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i+chunk_length_ms]
#         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(path, format="wav")
#         paths.append(path)
#     return paths


# # -------- TTS --------
# def chunked_tts(text, lang="en", max_chars=2000, out_path="outputs/summary_audio.mp3"):
#     os.makedirs("outputs", exist_ok=True)

#     if not text.strip():
#         return None

#     parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
#     segments = []

#     for i, p in enumerate(parts):
#         try:
#             temp = f"outputs/tmp_{i}.mp3"
#             gTTS(p, lang=lang).save(temp)
#             segments.append(AudioSegment.from_mp3(temp))
#         except:
#             continue

#     if not segments:
#         return None

#     audio = segments[0]
#     for s in segments[1:]:
#         audio += s

#     audio.export(out_path, format="mp3")
#     return out_path



# # -------- PIPELINE --------
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     logs = []

#     def log(msg):
#         logs.append(msg)
#         return "\n".join(logs)

#     try:
#         if not audio_path:
#             return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "")

#         log("⏳ Detecting language...")
#         detected_lang = transcriber.detect_language(audio_path)

#         log("🎙 Transcribing...")
#         chunks = chunk_audio_file(audio_path)
#         transcript = " ".join([
#             transcriber.transcribe(c, language=detected_lang).get("text", "")
#             for c in chunks
#         ]).strip()

#         if not transcript:
#             return (detected_lang, "", "", "", None, "", "", "", "❌ Empty transcript", "")

#         log("🧠 Running NLP...")
#         analysis = nlp.analyze(transcript, target_lang)

#         summary_raw = analysis.get("summary", [])
#         summary_text = " ".join(summary_raw) if isinstance(summary_raw, list) else str(summary_raw)
#         translated = analysis.get("translation", "") or summary_text
#         sentiment = analysis.get("sentiment", "neutral")

#         # -------- FIXED GA SECTION --------
#         log("🧬 Genetic Optimization...")

#         candidates = [
#             str(summary_text).strip(),
#             str(translated).strip(),
#             transcript[:800].strip()
#         ]

#         candidates = [c for c in candidates if isinstance(c, str) and len(c) > 10]

#         if len(candidates) < 2:
#             candidates.append(summary_text)
#         if len(candidates) < 2:
#             candidates.append(transcript[:500])

#         try:
#             best_summary, score = ga.evolve(candidates, transcript)
#         except:
#             best_summary = summary_text
#             score = 0.0

#         score_text = f"{round(max(score, 0.01) * 100, 2)}%"

#         # -------- AI Insight --------
#         log("💡 AI Insight...")
#         ai_text = ""
#         try:
#             ai_text = get_gemini_response(transcript) or ""
#         except:
#             pass

#         if custom_query.strip():
#             try:
#                 ans = get_gemini_response(custom_query)
#                 ai_text += f"\n\n📌 User Query:\n{custom_query}\n\n🧠 Answer:\n{ans}"
#             except:
#                 pass

#         # -------- TTS --------
#         log("🔊 Generating TTS...")
#         tts_lang = target_lang if target_lang in LANGS else "en"
#         tts_path = chunked_tts(best_summary, lang=tts_lang)

#         log("📤 Notifications...")
#         try: send_slack_message(best_summary)
#         except: pass

#         try:
#             rec = RECIPIENTS.copy() if RECIPIENTS else []
#             if extra_emails:
#                 rec.extend([e.strip() for e in extra_emails.split(",") if "@" in e])
#             if rec:
#                 send_email("Meeting Summary (GA Optimized)", best_summary, rec)
#         except:
#             pass

#         log("🎉 Done!")

#         return (
#             detected_lang,
#             transcript,
#             summary_text,
#             translated,
#             tts_path,
#             best_summary,
#             score_text,
#             sentiment,
#             "\n".join(logs),
#             ai_text
#         )

#     except Exception as e:
#         return ("", "", "", "", None, "", "", "", f"❌ Error:\n{traceback.format_exc()}", "")



# # -------- UI --------
# with gr.Blocks(css="footer{display:none}.gr-button{background:#2563eb;color:white;font-weight:bold;}") as ui:

#     gr.Markdown("## 🚀 AI Meeting Summarizer + GA Optimization")

#     with gr.Row():
#         with gr.Column(scale=1):
#             audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
#             lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Language")
#             custom_query = gr.Textbox(label="💬 Ask AI")
#             extra_emails = gr.Textbox(label="📧 Extra Emails")
#             btn = gr.Button("🔎 Process")

#         with gr.Column(scale=2):
#             status = gr.Markdown("⏳ Waiting...")

#             with gr.Tab("📄 Transcript"): out_lang = gr.Textbox(label="Detected Language"); out_trans = gr.Textbox(lines=8, show_copy_button=True)
#             with gr.Tab("📝 Summary"): out_sum = gr.Textbox(lines=10, show_copy_button=True)
#             with gr.Tab("🌐 Translation"): out_tr = gr.Textbox(lines=10, show_copy_button=True)
#             with gr.Tab("🔊 Audio Summary"): out_audio = gr.Audio(type="filepath")
#             with gr.Tab("🤖 GA Summary"): out_ga = gr.Textbox(lines=8, show_copy_button=True)
#             with gr.Tab("📈 GA Score"): out_score = gr.Textbox(label="Fitness %")
#             with gr.Tab("😊 Sentiment"): out_sent = gr.Textbox(label="Tone")
#             with gr.Tab("💡 AI Insight"): out_ai = gr.Textbox(lines=10, show_copy_button=True)

#     btn.click(
#         pipeline_with_status,
#         [audio_input, lang_input, custom_query, extra_emails],
#         [out_lang, out_trans, out_sum, out_tr, out_audio, out_ga, out_score, out_sent, status, out_ai]
#     )


# if __name__ == "__main__":
#     ui.launch()


# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment
# import traceback

# # 🧬 Genetic Algorithm Optimizer
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()

# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # ------------------ AUDIO CHUNKING ------------------
# def chunk_audio_file(audio_path, chunk_length_ms=60000):
#     """Split audio file into multiple .wav chunks."""
#     audio = AudioSegment.from_file(audio_path)
#     paths = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(path, format="wav")
#         paths.append(path)
#     return paths


# # ------------------ TTS ------------------
# def chunked_tts(text, lang="en", max_chars=2000):
#     """Text → MP3 safely even for long text."""
#     os.makedirs("outputs", exist_ok=True)
#     if not text or not str(text).strip():
#         return None

#     text = str(text)
#     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
#     segments = []

#     for idx, p in enumerate(parts):
#         try:
#             tmp = f"outputs/tts_{idx}.mp3"
#             gTTS(p, lang=lang).save(tmp)
#             segments.append(AudioSegment.from_mp3(tmp))
#         except Exception as e:
#             print("TTS error:", e)

#     if not segments:
#         return None

#     output_path = f"outputs/summary_audio_{datetime.now().strftime('%H%M%S')}.mp3"
#     final = segments[0]
#     for s in segments[1:]:
#         final += s

#     final.export(output_path, format="mp3")
#     return output_path


# # ------------------ MAIN PIPELINE ------------------
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     logs = []

#     def log(msg):
#         logs.append(msg)
#         return "\n".join(logs)

#     def normalize_output(data):
#         """Clean LLM output (list/dict/string) into readable bullet format."""
#         if isinstance(data, str):
#             return data.strip()

#         if isinstance(data, list):
#             return "\n".join(f"• {str(i).strip()}" for i in data)

#         if isinstance(data, dict):
#             formatted = []
#             for k, v in data.items():
#                 formatted.append(f"\n🔹 {k}:\n" if len(data) > 1 else "")
#                 if isinstance(v, list):
#                     formatted.extend(f"   • {item}" for item in v)
#                 else:
#                     formatted.append(f"   • {v}")
#             return "\n".join(formatted)

#         return str(data)

#     try:
#         if not audio_path:
#             return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "")

#         log("⏳ Detecting language...")
#         detected_lang = transcriber.detect_language(audio_path)

#         log("🎙 Transcribing (chunked)...")
#         chunks = chunk_audio_file(audio_path)
#         transcript = " ".join([
#             transcriber.transcribe(c, language=detected_lang).get("text", "")
#             for c in chunks
#         ]).strip()

#         if not transcript:
#             return (detected_lang, "", "", "", None, "", "", "", "❌ Transcript empty", "")

#         log("🧠 Running NLP summarization...")
#         analysis = nlp.analyze(transcript, target_lang)

#         # ------- CLEAN FORMATTING APPLIED -------
#         summary = normalize_output(analysis.get("summary", ""))

#         translation_raw = analysis.get("translation", "")
#         if isinstance(translation_raw, dict):
#             translated = translation_raw.get(target_lang) \
#                 or translation_raw.get("en") \
#                 or " ".join(str(v) for v in translation_raw.values())
#         elif isinstance(translation_raw, list):
#             translated = " ".join(str(item) for item in translation_raw)
#         else:
#             translated = str(translation_raw)

#         sentiment = normalize_output(analysis.get("sentiment", ""))

#         # ------- GA Optimization -------
#         log("🧬 Running Genetic Algorithm optimization...")

#         candidates = list({summary, translated, transcript[:800]})
#         candidates = [c for c in candidates if isinstance(c, str) and len(c.strip()) > 20]

#         if len(candidates) < 2:
#             candidates.append(transcript[:500])

#         try:
#             best_summary, score = ga.evolve(candidates, transcript)
#         except:
#             best_summary, score = summary, 0.0

#         score_text = f"{round(score * 100, 2)}%"

#         # ------- Gemini Insight -------
#         log("💡 Generating AI Insight...")
#         ai_msg = ""
#         try:
#             ai_msg = normalize_output(get_gemini_response(transcript)) or ""
#         except:
#             pass

#         if custom_query.strip():
#             try:
#                 reply = get_gemini_response(custom_query)
#                 ai_msg += f"\n\nQ: {custom_query}\nA: {normalize_output(reply)}"
#             except:
#                 pass

#         # ------- TTS -------
#         log("🔊 Creating Audio Summary...")
#         tts_lang = target_lang if target_lang in LANGS else "en"
#         tts_text = translated if translated and translated.strip() else best_summary
#         audio_out = chunked_tts(tts_text, lang=tts_lang)

#         # ------- Notifications -------
#         log("📤 Sending Notifications...")
#         try:
#             send_slack_message(best_summary)
#         except:
#             pass

#         if extra_emails:
#             try:
#                 emails = [e.strip() for e in extra_emails.split(",") if "@" in e]
#                 if emails:
#                     send_email("Meeting Summary (Optimized)", best_summary, emails)
#             except:
#                 pass

#         log("🎉 Done!")

#         return (
#             detected_lang,
#             transcript,
#             summary,
#             translated,
#             audio_out,
#             best_summary,
#             score_text,
#             sentiment,
#             "\n".join(logs),
#             ai_msg
#         )

#     except Exception as e:
#         err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
#         print(err)
#         return ("", "", "", "", None, "", "", "", err, "")


# # ------------------ UI ------------------
# with gr.Blocks(css="footer{visibility:hidden}") as ui:

#     gr.Markdown("## 🚀 AI Meeting Summarizer + Genetic Algorithm Optimization")

#     with gr.Row():
#         audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
#         lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#         custom_query = gr.Textbox(label="💬 Ask AI Insight")
#         extra_emails = gr.Textbox(label="📧 Extra Emails")
#         submit_btn = gr.Button("🔎 Process", variant="primary")

#     status_display = gr.Markdown("⏳ Waiting...")

#     with gr.Row():
#         with gr.Tab("📄 Transcript"):
#             out_lang = gr.Textbox(label="Language Detected", interactive=False)
#             out_text = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("📝 Summary"):
#             out_summary = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🌐 Translation"):
#             out_translated = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🔊 Audio"):
#             out_audio = gr.Audio(type="filepath")

#         with gr.Tab("🤖 GA Optimized Summary"):
#             out_ga = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("📈 GA Score"):
#             out_score = gr.Textbox(interactive=False)

#         with gr.Tab("😊 Sentiment"):
#             out_sentiment = gr.Textbox(interactive=False)

#         with gr.Tab("💡 AI Insight"):
#             out_ai = gr.Textbox(lines=8, show_copy_button=True)

#     submit_btn.click(
#         fn=pipeline_with_status,
#         inputs=[audio_input, lang_input, custom_query, extra_emails],
#         outputs=[
#             out_lang,        # 1
#             out_text,        # 2
#             out_summary,     # 3
#             out_translated,  # 4
#             out_audio,       # 5
#             out_ga,          # 6
#             out_score,       # 7
#             out_sentiment,   # 8
#             status_display,  # 9
#             out_ai           # 10
#         ]
#     )

# if __name__ == "__main__":
#     ui.launch()



# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment
# import traceback
# import matplotlib.pyplot as plt   # 🆕 Added

# # 🧬 Genetic Algorithm Optimizer
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()

# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # ------------------ AUDIO CHUNKING ------------------
# def chunk_audio_file(audio_path, chunk_length_ms=60000):
#     audio = AudioSegment.from_file(audio_path)
#     paths = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(path, format="wav")
#         paths.append(path)
#     return paths


# # ------------------ TTS ------------------
# def chunked_tts(text, lang="en", max_chars=2000):
#     os.makedirs("outputs", exist_ok=True)
#     if not text or not str(text).strip():
#         return None

#     text = str(text)
#     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
#     segments = []

#     for idx, p in enumerate(parts):
#         try:
#             tmp = f"outputs/tts_{idx}.mp3"
#             gTTS(p, lang=lang).save(tmp)
#             segments.append(AudioSegment.from_mp3(tmp))
#         except Exception as e:
#             print("TTS error:", e)

#     if not segments:
#         return None

#     output_path = f"outputs/summary_audio_{datetime.now().strftime('%H%M%S')}.mp3"
#     final = segments[0]
#     for s in segments[1:]:
#         final += s

#     final.export(output_path, format="mp3")
#     return output_path


# # ------------------ NEW: PLOT FUNCTION ------------------
# def generate_ga_graph(history):
#     """Create GA Convergence Graph from stored fitness history."""
#     if not history:
#         return None

#     os.makedirs("outputs", exist_ok=True)
#     path = "outputs/ga_convergence.png"

#     plt.figure(figsize=(6, 4))
#     plt.plot(history, marker="o")
#     plt.title("GA Convergence Graph")
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness Score")
#     plt.grid(True)
#     plt.savefig(path)
#     plt.close()

#     return path


# # ------------------ MAIN PIPELINE ------------------
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     logs = []

#     def log(msg):
#         logs.append(msg)
#         return "\n".join(logs)

#     def normalize_output(data):
#         if isinstance(data, str):
#             return data.strip()
#         if isinstance(data, list):
#             return "\n".join(f"• {str(i).strip()}" for i in data)
#         if isinstance(data, dict):
#             formatted = []
#             for k, v in data.items():
#                 formatted.append(f"\n🔹 {k}:\n" if len(data) > 1 else "")
#                 if isinstance(v, list):
#                     formatted.extend(f"   • {item}" for item in v)
#                 else:
#                     formatted.append(f"   • {v}")
#             return "\n".join(formatted)
#         return str(data)

#     try:
#         if not audio_path:
#             return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "", None)

#         log("⏳ Detecting language...")
#         detected_lang = transcriber.detect_language(audio_path)

#         log("🎙 Transcribing (chunked)...")
#         chunks = chunk_audio_file(audio_path)
#         transcript = " ".join([
#             transcriber.transcribe(c, language=detected_lang).get("text", "")
#             for c in chunks
#         ]).strip()

#         if not transcript:
#             return (detected_lang, "", "", "", None, "", "", "", "❌ Transcript empty", "", None)

#         log("🧠 Running NLP summarization...")
#         analysis = nlp.analyze(transcript, target_lang)

#         summary = normalize_output(analysis.get("summary", ""))

#         translation_raw = analysis.get("translation", "")
#         if isinstance(translation_raw, dict):
#             translated = translation_raw.get(target_lang) \
#                 or translation_raw.get("en") \
#                 or " ".join(str(v) for v in translation_raw.values())
#         elif isinstance(translation_raw, list):
#             translated = " ".join(str(item) for item in translation_raw)
#         else:
#             translated = str(translation_raw)

#         sentiment = normalize_output(analysis.get("sentiment", ""))

#         log("🧬 Running Genetic Algorithm optimization...")
#         candidates = list({summary, translated, transcript[:800]})
#         candidates = [c for c in candidates if isinstance(c, str) and len(c.strip()) > 20]

#         if len(candidates) < 2:
#             candidates.append(transcript[:500])

#         try:
#             best_summary, score = ga.evolve(candidates, transcript)
#             graph_path = generate_ga_graph(ga.history)  # 🆕 Create graph
#         except:
#             best_summary, score = summary, 0.0
#             graph_path = None

#         score_text = f"{round(score * 100, 2)}%"

#         log("💡 Generating AI Insight...")
#         ai_msg = ""
#         try:
#             ai_msg = normalize_output(get_gemini_response(transcript)) or ""
#         except:
#             pass

#         if custom_query.strip():
#             try:
#                 reply = get_gemini_response(custom_query)
#                 ai_msg += f"\n\nQ: {custom_query}\nA: {normalize_output(reply)}"
#             except:
#                 pass

#         log("🔊 Creating Audio Summary...")
#         tts_lang = target_lang if target_lang in LANGS else "en"
#         tts_text = translated if translated.strip() else best_summary
#         audio_out = chunked_tts(tts_text, lang=tts_lang)

#         log("📤 Sending Notifications...")
#         try:
#             send_slack_message(best_summary)
#         except:
#             pass

#         if extra_emails:
#             try:
#                 emails = [e.strip() for e in extra_emails.split(",") if "@" in e]
#                 if emails:
#                     send_email("Meeting Summary (Optimized)", best_summary, emails)
#             except:
#                 pass

#         log("🎉 Done!")

#         return (
#             detected_lang,
#             transcript,
#             summary,
#             translated,
#             audio_out,
#             best_summary,
#             score_text,
#             sentiment,
#             "\n".join(logs),
#             ai_msg,
#             graph_path        # 🆕 new return item
#         )

#     except Exception as e:
#         err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
#         print(err)
#         return ("", "", "", "", None, "", "", "", err, "", None)


# # ------------------ UI ------------------
# with gr.Blocks(css="footer{visibility:hidden}") as ui:

#     gr.Markdown("## 🚀 AI Meeting Summarizer + Genetic Algorithm Optimization")

#     with gr.Row():
#         audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
#         lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#         custom_query = gr.Textbox(label="💬 Ask AI Insight")
#         extra_emails = gr.Textbox(label="📧 Extra Emails")
#         submit_btn = gr.Button("🔎 Process", variant="primary")

#     status_display = gr.Markdown("⏳ Waiting...")

#     with gr.Row():
#         with gr.Tab("📄 Transcript"):
#             out_lang = gr.Textbox(label="Language Detected", interactive=False)
#             out_text = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("📝 Summary"):
#             out_summary = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🌐 Translation"):
#             out_translated = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🔊 Audio"):
#             out_audio = gr.Audio(type="filepath")

#         with gr.Tab("🤖 GA Optimized Summary"):
#             out_ga = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("📈 GA Score"):
#             out_score = gr.Textbox(interactive=False)

#         with gr.Tab("📉 GA Convergence Graph"):  # 🆕 new tab
#             out_graph = gr.Image(type="filepath")

#         with gr.Tab("😊 Sentiment"):
#             out_sentiment = gr.Textbox(interactive=False)

#         with gr.Tab("💡 AI Insight"):
#             out_ai = gr.Textbox(lines=8, show_copy_button=True)

#     submit_btn.click(
#         fn=pipeline_with_status,
#         inputs=[audio_input, lang_input, custom_query, extra_emails],
#         outputs=[
#             out_lang, out_text, out_summary, out_translated,
#             out_audio, out_ga, out_score, out_sentiment,
#             status_display, out_ai, out_graph  # 🆕 added graph output
#         ]
#     )

# if __name__ == "__main__":
#     ui.launch()


# import os
# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment
# import traceback
# import matplotlib.pyplot as plt   # Graph

# # 🧬 Genetic Algorithm Optimizer
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email, RECIPIENTS
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()

# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # ---------- AUDIO CHUNKING ----------
# def chunk_audio_file(audio_path, chunk_length_ms=60000):
#     audio = AudioSegment.from_file(audio_path)
#     paths = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(path, format="wav")
#         paths.append(path)
#     return paths


# # ---------- TTS ----------
# def chunked_tts(text, lang="en", max_chars=2000):
#     os.makedirs("outputs", exist_ok=True)
#     if not text or not str(text).strip():
#         return None

#     text = str(text)
#     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
#     segments = []

#     for idx, p in enumerate(parts):
#         try:
#             tmp = f"outputs/tts_{idx}.mp3"
#             gTTS(p, lang=lang).save(tmp)
#             segments.append(AudioSegment.from_mp3(tmp))
#         except Exception as e:
#             print("TTS error:", e)

#     if not segments:
#         return None

#     output_path = f"outputs/summary_audio_{datetime.now().strftime('%H%M%S')}.mp3"
#     final = segments[0]
#     for s in segments[1:]:
#         final += s

#     final.export(output_path, format="mp3")
#     return output_path


# # ---------- GA GRAPH ----------
# def generate_ga_graph(history):
#     if not history:
#         return None

#     os.makedirs("outputs", exist_ok=True)
#     path = "outputs/ga_convergence.png"

#     plt.figure(figsize=(6, 4))
#     plt.plot(history, marker="o")
#     plt.title("GA Convergence Graph")
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness Score")
#     plt.grid(True)
#     plt.savefig(path)
#     plt.close()

#     return path


# # ---------- MAIN PIPELINE ----------
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     logs = []

#     def log(msg):
#         logs.append(msg)
#         return "\n".join(logs)

#     def normalize_output(data):
#         if isinstance(data, str):
#             return data.strip()
#         if isinstance(data, list):
#             return "\n".join(f"• {str(i).strip()}" for i in data)
#         if isinstance(data, dict):
#             formatted = []
#             for k, v in data.items():
#                 formatted.append(f"\n🔹 {k}:\n")
#                 if isinstance(v, list):
#                     formatted.extend(f"   • {item}" for item in v)
#                 else:
#                     formatted.append(f"   • {v}")
#             return "\n".join(formatted)
#         return str(data)

#     try:
#         if not audio_path:
#             return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "", None, "")

#         log("⏳ Detecting language...")
#         detected_lang = transcriber.detect_language(audio_path)

#         log("🎙 Transcribing (chunked)...")
#         chunks = chunk_audio_file(audio_path)
#         transcript = " ".join([transcriber.transcribe(c, language=detected_lang).get("text", "") for c in chunks]).strip()

#         if not transcript:
#             return (detected_lang, "", "", "", None, "", "", "", "❌ Transcript empty", "", None, "")

#         log("🧠 Running NLP summarization...")
#         analysis = nlp.analyze(transcript, target_lang)

#         summary = normalize_output(analysis.get("summary", ""))

#         translation_raw = analysis.get("translation", "")
#         if isinstance(translation_raw, dict):
#             translated = translation_raw.get(target_lang) or translation_raw.get("en") or " ".join(str(v) for v in translation_raw.values())
#         elif isinstance(translation_raw, list):
#             translated = " ".join(str(item) for item in translation_raw)
#         else:
#             translated = str(translation_raw)

#         sentiment = normalize_output(analysis.get("sentiment", ""))

#         log("🧬 Running Genetic Algorithm optimization...")
#         candidates = list({summary, translated, transcript[:800]})
#         candidates = [c for c in candidates if isinstance(c, str) and len(c.strip()) > 20]

#         if len(candidates) < 2:
#             candidates.append(transcript[:500])

#         try:
#             best_summary, score = ga.evolve(candidates, transcript)
#             graph_path = generate_ga_graph(ga.history)
#         except:
#             best_summary, score = summary, 0.0
#             graph_path = None

#         score_text = f"{round(score * 100, 2)}%"

#         # -------- NEW: COMPARISON TABLE --------
#         comparison_table = (
#             "📊 **Baseline vs GA Optimized Summary**\n\n"
#             f"• Baseline Fitness: {0.01}\n"
#             f"• Optimized Fitness: {round(score, 4)}\n"
#             f"• Improvement: {round(score * 100, 2)}%\n"
#         )

#         log("💡 Generating AI Insight...")
#         ai_msg = ""
#         try:
#             ai_msg = normalize_output(get_gemini_response(transcript)) or ""
#         except:
#             pass

#         if custom_query.strip():
#             try:
#                 reply = get_gemini_response(custom_query)
#                 ai_msg += f"\n\nQ: {custom_query}\nA: {normalize_output(reply)}"
#             except:
#                 pass

#         log("🔊 Creating Audio Summary...")
#         tts_lang = target_lang if target_lang in LANGS else "en"
#         tts_text = translated if translated.strip() else best_summary
#         audio_out = chunked_tts(tts_text, lang=tts_lang)

#         log("📤 Sending Notifications...")
#         try: send_slack_message(best_summary)
#         except: pass

#         if extra_emails:
#             try:
#                 emails = [e.strip() for e in extra_emails.split(",") if "@" in e]
#                 if emails:
#                     send_email("Meeting Summary (Optimized)", best_summary, emails)
#             except:
#                 pass

#         log("🎉 Done!")

#         return (
#             detected_lang, transcript, summary, translated,
#             audio_out, best_summary, score_text, sentiment,
#             "\n".join(logs), ai_msg, graph_path, comparison_table
#         )

#     except Exception as e:
#         err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
#         return ("", "", "", "", None, "", "", "", err, "", None, "")


# # ---------- UI ----------
# with gr.Blocks(css="footer{visibility:hidden}") as ui:

#     gr.Markdown("## 🚀 AI Meeting Summarizer + Genetic Algorithm Optimization")

#     with gr.Row():
#         audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
#         lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#         custom_query = gr.Textbox(label="💬 Ask AI Insight")
#         extra_emails = gr.Textbox(label="📧 Extra Emails")
#         submit_btn = gr.Button("🔎 Process", variant="primary")

#     status_display = gr.Markdown("⏳ Waiting...")

#     with gr.Row():
#         with gr.Tab("📄 Transcript"): out_lang = gr.Textbox(interactive=False); out_text = gr.Textbox(lines=10, show_copy_button=True)
#         with gr.Tab("📝 Summary"): out_summary = gr.Textbox(lines=10, show_copy_button=True)
#         with gr.Tab("🌐 Translation"): out_translated = gr.Textbox(lines=10, show_copy_button=True)
#         with gr.Tab("🔊 Audio"): out_audio = gr.Audio(type="filepath")
#         with gr.Tab("🤖 GA Optimized Summary"): out_ga = gr.Textbox(lines=10, show_copy_button=True)
#         with gr.Tab("📈 GA Score"): out_score = gr.Textbox(interactive=False)
#         with gr.Tab("📊 Comparison Table"): out_compare = gr.Textbox(lines=8, interactive=False, show_copy_button=True)
#         with gr.Tab("📉 GA Convergence Graph"): out_graph = gr.Image(type="filepath")
#         with gr.Tab("😊 Sentiment"): out_sentiment = gr.Textbox(interactive=False)
#         with gr.Tab("💡 AI Insight"): out_ai = gr.Textbox(lines=8, show_copy_button=True)

#     submit_btn.click(
#         pipeline_with_status,
#         [audio_input, lang_input, custom_query, extra_emails],
#         [
#             out_lang, out_text, out_summary, out_translated,
#             out_audio, out_ga, out_score, out_sentiment,
#             status_display, out_ai, out_graph, out_compare
#         ]
#     )

# if __name__ == "__main__":
#     ui.launch()


import os
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# 🚀 Increase Gradio output handling capacity
os.environ["GRADIO_RESPONSE_STREAMING"] = "true"
os.environ["GRADIO_MAX_BODY_SIZE"] = "200000000"

from dotenv import load_dotenv
import gradio as gr
from gtts import gTTS
from datetime import datetime
from pydub import AudioSegment
import traceback
import matplotlib.pyplot as plt   # Graph

# 🧬 Genetic Algorithm Optimizer
from bio_algorithms.genetic_optimizer import GeneticOptimizer

# Agents & integrations
from agents.transcriber import TranscriberAgent
from agents.llm_nlp import LLMNLP
from agents.highlighter import HighlightAgent
from integrations.slack_notify import send_slack_message
from integrations.emailer import send_email, RECIPIENTS
from integrations.gemini_api import get_gemini_response

load_dotenv()

# Agents
transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
nlp = LLMNLP()
highlighter = HighlightAgent()
ga = GeneticOptimizer()

LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# ---------- AUDIO CHUNKING ----------
def chunk_audio_file(audio_path, chunk_length_ms=60000):
    audio = AudioSegment.from_file(audio_path)
    paths = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
        chunk.export(path, format="wav")
        paths.append(path)
    return paths


# ---------- TTS ----------
def chunked_tts(text, lang="en", max_chars=2000):
    os.makedirs("outputs", exist_ok=True)
    if not text or not str(text).strip():
        return None

    text = str(text)
    parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    segments = []

    for idx, p in enumerate(parts):
        try:
            tmp = f"outputs/tts_{idx}.mp3"
            gTTS(p, lang=lang).save(tmp)
            segments.append(AudioSegment.from_mp3(tmp))
        except Exception as e:
            print("TTS error:", e)

    if not segments:
        return None

    output_path = f"outputs/summary_audio_{datetime.now().strftime('%H%M%S')}.mp3"
    final = segments[0]
    for s in segments[1:]:
        final += s

    final.export(output_path, format="mp3")
    return output_path


# ---------- GA GRAPH ----------
def generate_ga_graph(history):
    if not history:
        return None

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/ga_convergence.png"

    plt.figure(figsize=(6, 4))
    plt.plot(history, marker="o")
    plt.title("GA Convergence Graph")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    return path


# ---------- MAIN PIPELINE ----------
def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
    logs = []

    def log(msg):
        logs.append(msg)
        return "\n".join(logs[-50:])  # limit display to last 50 logs

    def normalize_output(data):
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, list):
            return "\n".join(f"• {str(i).strip()}" for i in data)
        if isinstance(data, dict):
            formatted = []
            for k, v in data.items():
                formatted.append(f"\n🔹 {k}:\n")
                if isinstance(v, list):
                    formatted.extend(f"   • {item}" for item in v)
                else:
                    formatted.append(f"   • {v}")
            return "\n".join(formatted)
        return str(data)

    try:
        if not audio_path:
            return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "", None, "")

        log("⏳ Detecting language...")
        detected_lang = transcriber.detect_language(audio_path)

        log("🎙 Transcribing (chunked)...")
        chunks = chunk_audio_file(audio_path)
        transcript = " ".join([transcriber.transcribe(c, language=detected_lang).get("text", "") for c in chunks]).strip()

        if not transcript:
            return (detected_lang, "", "", "", None, "", "", "", "❌ Transcript empty", "", None, "")

        # SAVE TRANSCRIPT AS FILE FOR LARGE INPUTS
        transcript_path = "outputs/transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        log("🧠 Running NLP summarization...")
        analysis = nlp.analyze(transcript, target_lang)

        summary = normalize_output(analysis.get("summary", ""))

        translation_raw = analysis.get("translation", "")
        if isinstance(translation_raw, dict):
            translated = translation_raw.get(target_lang) or translation_raw.get("en") or " ".join(str(v) for v in translation_raw.values())
        elif isinstance(translation_raw, list):
            translated = " ".join(str(item) for item in translation_raw)
        else:
            translated = str(translation_raw)

        sentiment = normalize_output(analysis.get("sentiment", ""))

        log("🧬 Running Genetic Algorithm optimization...")
        candidates = list({summary, translated, transcript[:800]})
        candidates = [c for c in candidates if isinstance(c, str) and len(c.strip()) > 20]

        if len(candidates) < 2:
            candidates.append(transcript[:500])

        try:
            best_summary, score = ga.evolve(candidates, transcript)
            graph_path = generate_ga_graph(ga.history)
        except:
            best_summary, score = summary, 0.0
            graph_path = None

        score_text = f"{round(score * 100, 2)}%"

        comparison_table = (
            "📊 **Baseline vs GA Optimized Summary**\n\n"
            f"• Baseline Fitness: {0.01}\n"
            f"• Optimized Fitness: {round(score, 4)}\n"
            f"• Improvement: {round(score * 100, 2)}%\n"
        )

        log("💡 Generating AI Insight...")
        ai_msg = ""
        try:
            ai_msg = normalize_output(get_gemini_response(transcript)) or ""
        except:
            pass

        if custom_query.strip():
            try:
                reply = get_gemini_response(custom_query)
                ai_msg += f"\n\nQ: {custom_query}\nA: {normalize_output(reply)}"
            except:
                pass

        log("🔊 Creating Audio Summary...")
        tts_lang = target_lang if target_lang in LANGS else "en"
        tts_text = translated if translated.strip() else best_summary
        audio_out = chunked_tts(tts_text, lang=tts_lang)

        log("📤 Sending Notifications...")
        try: send_slack_message(best_summary)
        except: pass

        if extra_emails:
            try:
                emails = [e.strip() for e in extra_emails.split(",") if "@" in e]
                if emails:
                    send_email("Meeting Summary (Optimized)", best_summary, emails)
            except:
                pass

        log("🎉 Done!")

        return (
            detected_lang, transcript_path, summary, translated,
            audio_out, best_summary, score_text, sentiment,
            "\n".join(logs[-50:]), ai_msg, graph_path, comparison_table
        )

    except Exception as e:
        err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
        return ("", "", "", "", None, "", "", "", err, "", None, "")

# ---------- UI ----------
with gr.Blocks(css="footer{visibility:hidden}") as ui:

    gr.Markdown("## 🚀 AI Meeting Summarizer + Genetic Algorithm Optimization")

    with gr.Row():
        audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
        lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
        custom_query = gr.Textbox(label="💬 Ask AI Insight")
        extra_emails = gr.Textbox(label="📧 Extra Emails")
        submit_btn = gr.Button("🔎 Process", variant="primary")

    status_display = gr.Markdown("⏳ Waiting...")

    with gr.Row():
        with gr.Tab("📄 Transcript File Download"): 
            out_lang = gr.Textbox(interactive=False)
            out_text = gr.File(label="📄 Download Transcript")

        with gr.Tab("📝 Summary"):
            out_summary = gr.Textbox(lines=10, show_copy_button=True)

        with gr.Tab("🌐 Translation"):
            out_translated = gr.Textbox(lines=10, show_copy_button=True)

        with gr.Tab("🔊 Audio"):
            out_audio = gr.Audio(type="filepath")

        with gr.Tab("🤖 GA Optimized Summary"):
            out_ga = gr.Textbox(lines=10, show_copy_button=True)

        # ✅ MERGED GA TAB
        with gr.Tab("📈 GA Performance"):
            out_score = gr.Textbox(label="GA Score", interactive=False)
            out_compare = gr.Textbox(label="Baseline vs Optimized Comparison", lines=8, interactive=False, show_copy_button=True)

        # 🔥 GA Graph Separate
        with gr.Tab("📉 GA Convergence Graph"):
            out_graph = gr.Image(type="filepath")

        with gr.Tab("😊 Sentiment"):
            out_sentiment = gr.Textbox(interactive=False)

        with gr.Tab("💡 AI Insight"):
            out_ai = gr.Textbox(lines=8, show_copy_button=True)

    submit_btn.click(
        pipeline_with_status,
        [audio_input, lang_input, custom_query, extra_emails],
        [
            out_lang, out_text, out_summary, out_translated,
            out_audio, out_ga, out_score, out_sentiment,
            status_display, out_ai, out_graph, out_compare
        ]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    ui.launch(
        server_name="0.0.0.0",
        server_port=port
    )





# import os
# import warnings
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# # 🚀 Increase Gradio output handling capacity
# os.environ["GRADIO_RESPONSE_STREAMING"] = "true"
# os.environ["GRADIO_MAX_BODY_SIZE"] = "200000000"

# from dotenv import load_dotenv
# import gradio as gr
# from gtts import gTTS
# from datetime import datetime
# from pydub import AudioSegment
# import traceback
# import matplotlib.pyplot as plt   # Graph

# # 🧬 Genetic Algorithm Optimizer
# from bio_algorithms.genetic_optimizer import GeneticOptimizer

# # Agents & integrations
# from agents.transcriber import TranscriberAgent
# from agents.llm_nlp import LLMNLP
# from agents.highlighter import HighlightAgent
# from integrations.slack_notify import send_slack_message
# from integrations.emailer import send_email
# from integrations.gemini_api import get_gemini_response

# load_dotenv()

# # Agents
# transcriber = TranscriberAgent(model_name=os.getenv("WHISPER_MODEL", "base"))
# nlp = LLMNLP()
# highlighter = HighlightAgent()
# ga = GeneticOptimizer()

# LANGS = ["hi", "ta", "kn", "te", "bn", "fr", "es", "en"]
# DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")


# # ---------- AUDIO CHUNKING ----------
# def chunk_audio_file(audio_path, chunk_length_ms=60000):
#     audio = AudioSegment.from_file(audio_path)
#     paths = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         path = f"{audio_path}_chunk_{i // chunk_length_ms}.wav"
#         chunk.export(path, format="wav")
#         paths.append(path)
#     return paths


# # ---------- TTS ----------
# def chunked_tts(text, lang="en", max_chars=2000):
#     os.makedirs("outputs", exist_ok=True)
#     if not text or not str(text).strip():
#         return None

#     text = str(text)
#     parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
#     segments = []

#     for idx, p in enumerate(parts):
#         try:
#             tmp = f"outputs/tts_{idx}.mp3"
#             gTTS(p, lang=lang).save(tmp)
#             segments.append(AudioSegment.from_mp3(tmp))
#         except Exception:
#             pass

#     if not segments:
#         return None

#     output_path = f"outputs/summary_audio_{datetime.now().strftime('%H%M%S')}.mp3"
#     final = segments[0]
#     for s in segments[1:]:
#         final += s

#     final.export(output_path, format="mp3")
#     return output_path


# # ---------- NEW: BASELINE vs GA GRAPH ----------
# def generate_ga_graph(history, baseline=0.01):
#     """📉 Create line chart comparing baseline vs GA optimized evolution."""
#     if not history:
#         return None

#     os.makedirs("outputs", exist_ok=True)
#     path = "outputs/ga_performance_graph.png"

#     generations = list(range(len(history)))
#     baseline_line = [baseline] * len(history)

#     plt.figure(figsize=(6, 4))
    
#     plt.plot(generations, baseline_line, linestyle="--", marker="x", label="Baseline Fitness")
#     plt.plot(generations, history, linestyle="-", marker="o", label="GA Optimized Fitness")

#     plt.title("📈 Baseline vs GA Optimization Performance")
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness Score")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(path)
#     plt.close()

#     return path


# # ---------- MAIN PIPELINE ----------
# def pipeline_with_status(audio_path, target_lang, custom_query, extra_emails):
#     logs = []

#     def log(msg):
#         logs.append(msg)
#         return "\n".join(logs[-50:])

#     def normalize_output(data):
#         if isinstance(data, str): return data.strip()
#         if isinstance(data, list): return "\n".join(f"• {str(i).strip()}" for i in data)
#         if isinstance(data, dict):
#             formatted = []
#             for k, v in data.items():
#                 formatted.append(f"\n🔹 {k}:\n")
#                 if isinstance(v, list): formatted.extend(f"   • {item}" for item in v)
#                 else: formatted.append(f"   • {v}")
#             return "\n".join(formatted)
#         return str(data)

#     try:
#         if not audio_path:
#             return ("", "", "", "", None, "", "", "", "❌ No audio uploaded", "", None, "")

#         log("⏳ Detecting language...")
#         detected_lang = transcriber.detect_language(audio_path)

#         log("🎙 Transcribing...")
#         chunks = chunk_audio_file(audio_path)
#         transcript = " ".join([transcriber.transcribe(c, language=detected_lang).get("text", "") for c in chunks]).strip()

#         transcript_path = "outputs/transcript.txt"
#         with open(transcript_path, "w", encoding="utf-8") as f:
#             f.write(transcript)

#         log("🧠 NLP Summarizing...")
#         analysis = nlp.analyze(transcript, target_lang)

#         summary = normalize_output(analysis.get("summary", ""))

#         translation_raw = analysis.get("translation", "")
#         translated = (translation_raw.get(target_lang) if isinstance(translation_raw, dict) else translation_raw)

#         sentiment = normalize_output(analysis.get("sentiment", ""))

#         log("🧬 Running Genetic Algorithm...")
#         candidates = list({summary, translated, transcript[:800]})
#         candidates = [c for c in candidates if isinstance(c, str) and len(c.strip()) > 20]

#         try:
#             best_summary, score = ga.evolve(candidates, transcript)
#             graph_path = generate_ga_graph(ga.history, baseline=0.01)
#         except:
#             best_summary, score = summary, 0.0
#             graph_path = None

#         score_text = f"{round(score * 100, 2)}%"

#         comparison_table = (
#             "📊 **Baseline vs GA Optimized Summary**\n\n"
#             f"• Baseline Fitness: 0.01\n"
#             f"• Optimized Fitness: {round(score, 4)}\n"
#             f"• Improvement: {round(score * 100, 2)}%\n"
#         )

#         log("💡 AI Insight...")
#         ai_msg = normalize_output(get_gemini_response(transcript)) if custom_query.strip() == "" else ""

#         log("🔊 Creating Audio...")
#         audio_out = chunked_tts(best_summary, target_lang)

#         log("🎉 Done!")

#         return (
#             detected_lang, transcript_path, summary, translated,
#             audio_out, best_summary, score_text, sentiment,
#             "\n".join(logs[-50:]), ai_msg, graph_path, comparison_table
#         )

#     except Exception as e:
#         return ("", "", "", "", None, "", "", "", f"❌ Error: {e}", "", None, "")


# # ---------------- UI ----------------
# with gr.Blocks(css="footer{visibility:hidden}") as ui:

#     gr.Markdown("## 🚀 AI Meeting Summarizer + Genetic Algorithm Optimization")

#     with gr.Row():
#         audio_input = gr.Audio(label="🎙 Upload Audio", type="filepath")
#         lang_input = gr.Dropdown(LANGS, value=DEFAULT_LANG, label="🌍 Target Language")
#         custom_query = gr.Textbox(label="💬 Ask AI Insight")
#         extra_emails = gr.Textbox(label="📧 Extra Emails")
#         submit_btn = gr.Button("🔎 Process", variant="primary")

#     status_display = gr.Markdown("⏳ Waiting...")

#     with gr.Row():

#         with gr.Tab("📄 Transcript File Download"):
#             out_lang = gr.Textbox(interactive=False)
#             out_text = gr.File(label="📄 Download Transcript")

#         with gr.Tab("📝 Summary"):
#             out_summary = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🌐 Translation"):
#             out_translated = gr.Textbox(lines=10, show_copy_button=True)

#         with gr.Tab("🔊 Audio"):
#             out_audio = gr.Audio(type="filepath")

#         with gr.Tab("🤖 GA Optimized Summary"):
#             out_ga = gr.Textbox(lines=10, show_copy_button=True)

#         # 📌 Combined GA Tab
#         with gr.Tab("📈 GA Performance"):
#             out_score = gr.Textbox(label="GA Score", interactive=False)
#             out_compare = gr.Textbox(label="Baseline vs Optimized Comparison", lines=8, interactive=False, show_copy_button=True)
#             out_graph = gr.Image(label="📉 Performance Line Graph", type="filepath")

#         # Original convergence stays
#         with gr.Tab("📉 GA Convergence Graph"):
#             out_graph_old = gr.Image(type="filepath")

#         with gr.Tab("😊 Sentiment"):
#             out_sentiment = gr.Textbox(interactive=False)

#         with gr.Tab("💡 AI Insight"):
#             out_ai = gr.Textbox(lines=8, show_copy_button=True)


#     submit_btn.click(
#         pipeline_with_status,
#         [audio_input, lang_input, custom_query, extra_emails],
#         [
#             out_lang, out_text, out_summary, out_translated,
#             out_audio, out_ga, out_score, out_sentiment,
#             status_display, out_ai, out_graph, out_compare
#         ]
#     )

# if __name__ == "__main__":
#     ui.launch()

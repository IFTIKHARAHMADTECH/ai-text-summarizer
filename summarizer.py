from transformers import pipeline
import gradio as gr

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if not text.strip():
        return "Please enter some text!"
    
    words = text.split()
    chunks = [" ".join(words[i:i+300]) for i in range(0, len(words), 300)]
    
    final_summary = []
    for chunk in chunks:
        summary = summarizer(chunk, max_new_tokens=60, do_sample=False)
        final_summary.append(summary[0]['summary_text'])
    
    return " ".join(final_summary)

iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=12, 
        placeholder="Paste your text here...", 
        label="Enter Text to Summarize"
    ),
    outputs=gr.Textbox(
        label="Summary",
        lines=8
    ),
    title="📝 AI Text Summarizer",
    description="Paste any text below and click 'Submit' to get a clear, concise summary instantly.",
    theme="default", 
    allow_flagging="never",
    live=False
)

iface.launch()

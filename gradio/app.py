import traceback

import gradio as gr
import requests

from deployment.gradio.client import analyze_video, ask_question


def run_analysis(video_file, topk: int, language: str):
    if video_file is None:
        return "Please upload a video first.", "", [], "", "", {}, {}

    video_path = video_file
    if hasattr(video_file, "name"):
        video_path = video_file.name

    try:
        summary, mapping, rows, explanation_text, explanation_meta, prediction, payload = analyze_video(
            video_path=video_path,
            topk=int(topk),
            language=language,
        )
        return summary, mapping, rows, explanation_text, explanation_meta, prediction, payload
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text if exc.response is not None else str(exc)
        return f"API HTTP error:\n{detail}", "", [], "", "", {}, {}
    except Exception as exc:
        return f"Run failed:\n{exc}\n\n{traceback.format_exc()}", "", [], "", "", {}, {}


def run_qa(prediction_state, question: str, language: str):
    try:
        answer_text, answer_meta, payload = ask_question(
            prediction=prediction_state,
            question=question,
            language=language,
        )
        return answer_text, answer_meta, payload
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text if exc.response is not None else str(exc)
        return f"API HTTP error:\n{detail}", "", {}
    except Exception as exc:
        return f"Run failed:\n{exc}\n\n{traceback.format_exc()}", "", {}


def fill_question_why():
    return "Why is this classified as the predicted action?"


def fill_question_not_second():
    return "Why not the second-ranked class?"


def fill_question_reliable():
    return "Is this result reliable?"


with gr.Blocks(title="SemSparse Video Demo") as demo:
    gr.Markdown(
        """
        # SemSparse Video Action Demo
        Upload a demo video, then call `/analyze/video-demo` from FastAPI.
        This demo maps the uploaded video filename to a prebuilt skeleton sample and generates an explanation from an LLM or template fallback.
        """
    )

    with gr.Row():
        video_input = gr.File(label="Upload Video (.mp4/.avi)", file_types=[".mp4", ".avi"])
        topk_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K")
        language_input = gr.Dropdown(choices=["zh", "en"], value="zh", label="Language")

    analyze_btn = gr.Button("Analyze")
    prediction_state = gr.State({})

    summary_output = gr.Textbox(label="Prediction Summary", lines=8)
    mapping_output = gr.Textbox(label="Mapping Details", lines=7)
    topk_output = gr.Dataframe(
        headers=["rank", "label_id", "class_name", "score"],
        datatype=["number", "number", "str", "number"],
        label="Top-K",
        wrap=True,
    )
    explanation_output = gr.Textbox(label="Explanation", lines=8)
    explanation_meta_output = gr.Textbox(label="Explanation Meta", lines=2)

    gr.Markdown("### Follow-up QA")
    question_input = gr.Textbox(label="Ask a Question", lines=2, placeholder="Why is this classified as cheer up?")
    with gr.Row():
        quick_q1 = gr.Button("Why this class?")
        quick_q2 = gr.Button("Why not the 2nd class?")
        quick_q3 = gr.Button("Is this result reliable?")
    ask_btn = gr.Button("Ask")
    qa_output = gr.Textbox(label="QA Answer", lines=6)
    qa_meta_output = gr.Textbox(label="QA Meta", lines=2)
    qa_json_output = gr.JSON(label="Raw QA Response")
    json_output = gr.JSON(label="Raw API Response")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[video_input, topk_input, language_input],
        outputs=[
            summary_output,
            mapping_output,
            topk_output,
            explanation_output,
            explanation_meta_output,
            prediction_state,
            json_output,
        ],
    )

    quick_q1.click(fn=fill_question_why, outputs=question_input)
    quick_q2.click(fn=fill_question_not_second, outputs=question_input)
    quick_q3.click(fn=fill_question_reliable, outputs=question_input)

    ask_btn.click(
        fn=run_qa,
        inputs=[prediction_state, question_input, language_input],
        outputs=[qa_output, qa_meta_output, qa_json_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

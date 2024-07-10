import itertools
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from pydantic import BaseModel
import PyPDF2
from fpdf import FPDF
import json

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.chat.chat_service import ChatService, ChatMessage, MessageRole, CompletionGen
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "Hydac SecChain GPT"
SOURCES_SEPARATOR = "\n\n Sources: \n"

class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chunks_service: ChunksService,
        chat_service: ChatService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chunks_service = chunks_service
        self._chat_service = chat_service

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None
        self.questions = []
        self.answers = []

    def _parse_pdf(self, file: Any) -> str:
        if file is None or not file.name.endswith('.pdf'):
            return "Please only upload PDF files."

        # Parse the PDF
        with open(file.name, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()

        return text

    def yield_deltas(self, completion_gen: CompletionGen) -> Iterable[str]:
        full_response: str = ""
        stream = completion_gen.response
        for delta in stream:
            if isinstance(delta, str):
                full_response += str(delta)
            elif isinstance(delta, ChatResponse):
                full_response += delta.delta or ""
            yield full_response
            time.sleep(0.02)

        if completion_gen.sources:
            full_response += SOURCES_SEPARATOR
            cur_sources = Source.curate_sources(completion_gen.sources)
            sources_text = "\n\n\n"
            used_files = set()
            for index, source in enumerate(cur_sources, start=1):
                if f"{source.file}-{source.page}" not in used_files:
                    sources_text = (
                        sources_text
                        + f"{index}. {source.file} (page {source.page}) \n\n"
                    )
                    used_files.add(f"{source.file}-{source.page}")
            full_response += sources_text
        yield full_response

    def _extract_questions(self, text: str) -> list[str]:
        logger.info('Extracted questions from text: %s', text)
        # Use the local chat service to extract questions
        prompt = "Extract all questions from the following text, return a json string of the question list. Example: ['sind Richtlinien zur Informationssicherheit vorhanden?','werden Informationssicherheitsrisiken gemanagt?']"
        messages = [
            ChatMessage(content=prompt, role=MessageRole.ASSISTANT),
            ChatMessage(content=text, role=MessageRole.USER)
        ]
        completion = self._chat_service.chat(messages=messages, use_context=False)
        logger.info('Extracted questions from chatmessage: %s', messages[1])
        extracted_questions = completion.response
        logger.info('Extracted questions: %s', extracted_questions)
        return json.loads(extracted_questions)  # Remove duplicates

    def _answer_question(self, question: str) -> Iterable[str]:
        prompt = '''
    Instruction: You are an IT security expert for a famous German company called Hydac.
    Given the question and related context, generate a short, clear, and professional response in German.
    You can refer to the example give below -----
    Question: Inwieweit werden alle Mitarbeiter zur Einhaltung der Informationssicherheit verpflichtet?
    Context: 2.1.2 Inwieweit werden alle Mitarbeiter zur Einhaltung der Informationssicherheit verpflichtet?
    Detaillierte Sachverhaltsdarstellung (inkl. Beurteilungsverfahren) Betrachtete Dokumente/Nachweise/Prüfungshandlung:
    Datengeheimnis Arbeitsvertrag,  IT-SRL, Vorlagen  Datenschutz Statement Hydac: Im Arbeitsvertrag wird auf die Einhaltung
    von Verfahrensanweisungen und im Unternehmen geltenden Richtlinien sowie Geheimhaltung verpflichtet. Mit Abschluss des Arbeitsvertrages 
    erhält der Mitarbeiter eine Erklärung zum Datengeheimnis, die er mit dem Unterzeichnen einwilligt. Weitere Verpflichtungen zur Geheimhaltung 
    können je nach Berufsgruppe bestehen und werden durch den Zentralbereich Datenschutz ausgegeben. Die IT-SRL ist eine Verfahrensanweisung 
    und somit für alle Mitarbeiter bindend. Mitarbeiter werden zur Geheimhaltung und auf das Regelwerk zur Informationssicherheit verpflichtet. 
    Feststellung Auf Basis der Beobachtungen wurde keine Abweichung festgestellt. 
    Answer: Im Arbeitsvertrag wird auf die Einhaltung von Verfahrensanweisungen und im Unternehmen geltenden Richtlinien sowie Geheimhaltung 
    verpflichtet. Mit Abschluss des Arbeitsvertrages erhält der Mitarbeiter eine Erklärung zum Datengeheimnis, die er mit dem Unterzeichnen einwilligt. 
    Weitere Verpflichtungen zur Geheimhaltung können je nach Berufsgruppe bestehen und werden durch den Zentralbereich Datenschutz ausgegeben. 
    Die IT-SRL ist eine Verfahrensanweisung und somit für alle Mitarbeiter bindend Mitarbeiter werden zur Geheimhaltung und auf das Regelwerk zur Informationssicherheit verpflichtet..
'''
        messages = [
            ChatMessage(content=prompt, role=MessageRole.ASSISTANT),
            ChatMessage(content=question, role=MessageRole.USER)
        ]
        completion = self._chat_service.chat(messages=messages, use_context=True)
        return completion.response

    def _process_upload(self, file: Any) -> str:
        # Check file format
        if not file.name.endswith('.pdf'):
            return "Please only upload PDF files."

        # Parse the PDF to extract text
        text = self._parse_pdf(file)

        # Extract questions using the local chat service
        self.questions = self._extract_questions(text)
        
        return "\n".join(self.questions)

    def _answer_questionnaire(self) -> str:
        self.answers = []
        for question in self.questions:
            answer = list(self._answer_question(question))
            self.answers.append((question, answer))
        return "\n".join([f"Q: {q}\nA: {''.join(a)}\n\n" for q, a in self.answers])

    def _generate_pdf(self) -> str:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for question, answer in self.answers:
            pdf.multi_cell(0, 10, f"Q: {question}\nA: {''.join(answer)}\n\n")
        output_path = "/mnt/data/filled_questionnaire.pdf"
        pdf.output(output_path)
        return output_path

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[Any]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file.name) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #A03232;"  # Change header color
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "color: white;"  # Change font color to white
            "}"
            ".logo img { height: 0% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'><img src={logo_svg} alt='Hydac SecChain GPT'><h1 style='color: white;'>Hydac SecChain GPT</h1></div>")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=settings().ui.delete_file_button_enabled,
                        interactive=False,
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=settings().ui.delete_all_files_button_enabled,
                    )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )

                with gr.Column(scale=7, elem_id="col"):
                    upload_questionnaire_button = gr.File(label="Upload Questionnaire PDF", type="filepath")
                    questionnaire_output = gr.Textbox(label="Extracted Questions", lines=20)
                    answer_button = gr.Button("Answer Questionnaire")
                    download_button = gr.Button("Download Filled Questionnaire", visible=False)
                    download_link = gr.File()

                    upload_questionnaire_button.upload(fn=self._process_upload, inputs=upload_questionnaire_button, outputs=questionnaire_output)
                    answer_button.click(fn=self._answer_questionnaire, outputs=questionnaire_output)
                    download_button.click(fn=self._generate_pdf, outputs=download_link)

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)

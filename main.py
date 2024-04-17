import json
import os
import time

import gradio as gr
import oci
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

compartment_id = os.environ["MY_COMPARTMENT_OCID"]

config = oci.config.from_file()
config.update({"region": "ap-tokyo-1"})

bucket_name = 'poc-bucket'
object_name = 'output.wav'

# Initialize client
ai_language_client = oci.ai_language.AIServiceLanguageClient(config)
object_storage_client = oci.object_storage.ObjectStorageClient(config)
namespace = object_storage_client.get_namespace().data


def clear_chat(chat_history):
    chat_history = ""
    return chat_history


def recognize(chat_history, user_or_support_radio_input, audio_microphone_input):
    if audio_microphone_input:
        # Upload the file
        with open(audio_microphone_input, 'rb') as f:
            object_storage_client.put_object(namespace, bucket_name, object_name, f)
        print("File uploaded successfully!")

        # Recognize the file
        object_names = [object_name]
        input_object_location = oci.ai_speech.models.ObjectLocation(namespace_name=namespace, bucket_name=bucket_name,
                                                                    object_names=object_names)
        input_location = oci.ai_speech.models.ObjectListInlineInputLocation(
            location_type="OBJECT_LIST_INLINE_INPUT_LOCATION", object_locations=[input_object_location])
        ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config)
        create_transcription_job_response = ai_speech_client.create_transcription_job(
            create_transcription_job_details=oci.ai_speech.models.CreateTranscriptionJobDetails(
                compartment_id=compartment_id,
                input_location=input_location,
                output_location=oci.ai_speech.models.OutputLocation(
                    namespace_name=namespace,
                    bucket_name=bucket_name,
                    prefix="speech-transcription"),
                additional_transcription_formats=["SRT"],
                model_details=oci.ai_speech.models.TranscriptionModelDetails(
                    model_type="WHISPER_MEDIUM",
                    domain="GENERIC",
                    language_code="ja",
                    transcription_settings=oci.ai_speech.models.TranscriptionSettings(
                        diarization=oci.ai_speech.models.Diarization(
                            is_diarization_enabled=False))),
                normalization=oci.ai_speech.models.TranscriptionNormalization(
                    is_punctuation_enabled=True),
            )
        )
        create_transcription_job_id = create_transcription_job_response.data.id
        get_transcription_job_response = ai_speech_client.get_transcription_job(
            transcription_job_id=create_transcription_job_id)
        lifecycle_state = get_transcription_job_response.data.lifecycle_state
        output_location = get_transcription_job_response.data.output_location
        get_transcription_timeout = 60
        get_transcription_sleep_time = 0
        while lifecycle_state != 'SUCCEEDED':
            if get_transcription_sleep_time > get_transcription_timeout:
                break
            get_transcription_job_response = ai_speech_client.get_transcription_job(
                transcription_job_id=create_transcription_job_id)
            lifecycle_state = get_transcription_job_response.data.lifecycle_state
            if lifecycle_state == 'SUCCEEDED':
                break
            time.sleep(1)
            get_transcription_sleep_time += 1

        get_object_response = object_storage_client.get_object(
            namespace_name=output_location.namespace_name,
            bucket_name=output_location.bucket_name,
            object_name=output_location.prefix + output_location.namespace_name + '_' + output_location.bucket_name + '_' + object_name + '.json')
        json_data = get_object_response.data.content.decode('utf-8')
        # print(f"{json_data=}")
        dict_data = json.loads(json_data)
        # print(f"{dict_data=}")
        sentences = dict_data['transcriptions']
        voice_text = ""
        for sentence in sentences:
            voice_text += sentence['transcription']
        print(f"{voice_text=}")

        # Translate the text
        batch_language_translation_response = ai_language_client.batch_language_translation(
            batch_language_translation_details=oci.ai_language.models.BatchLanguageTranslationDetails(
                documents=[
                    oci.ai_language.models.TextDocument(
                        key="translated_text",
                        text=voice_text,
                        language_code="ja")],
                compartment_id=compartment_id,
                target_language_code="en"))
        # print(f"{batch_language_translation_response.data=}")
        translated_text = batch_language_translation_response.data.documents[0].translated_text
        print(f"{translated_text=}")

        if user_or_support_radio_input == "User":
            chat_history = chat_history + [(None, voice_text + "\n[Translation] " + translated_text)]
        elif user_or_support_radio_input == "Support":
            chat_history = chat_history + [(voice_text + "\n[Translation] " + translated_text, None)]

        return chat_history, gr.Audio(value=None), gr.Textbox(value=translated_text)

    return chat_history, gr.Audio(), gr.Textbox()


def sentiment_analysis(text_output_input):
    batch_detect_language_sentiments_response = ai_language_client.batch_detect_language_sentiments(
        batch_detect_language_sentiments_details=oci.ai_language.models.BatchDetectLanguageSentimentsDetails(
            documents=[
                oci.ai_language.models.TextDocument(
                    key="sentiment_text",
                    text=text_output_input,
                    language_code="en")],
            compartment_id=compartment_id),
        level=["SENTENCE"])

    # Get the data from response
    # print(f"{batch_detect_language_sentiments_response.data=}")
    sentiment_scores = batch_detect_language_sentiments_response.data.documents[0].sentences[0].scores
    print(f"{sentiment_scores=}")


common_css = """
.gradio-container-3-25-0 {
  font-family: Arial, sans-serif;
}

main svelte-1kyws56 {
  width: 100%;
} 

footer > .svelte-1ax1toq {
  visibility: hidden;
}
"""

with gr.Blocks(title="Support System", css=common_css) as app:
    with gr.Row():
        with gr.Column(scale=20, min_width=0):
            support_chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History", show_label=True,
                                         show_copy_button=False, height=656)
        with gr.Column(scale=10, min_width=0):
            summary_textbox = gr.Textbox(lines=30, show_label=False, info="Summary")
    with gr.Row():
        user_or_support_radio = gr.Radio(label="User of Support", show_label=False,
                                         choices=[("User", "User"),
                                                  ("Support", "Support")],
                                         value="Support", interactive=True)
    with gr.Row():
        audio_microphone = gr.Microphone(label="Voice Input", type="filepath",
                                         show_edit_button=False)
    with gr.Row(visible=False):
        text_output = gr.Textbox()
    with gr.Row():
        with gr.Column(scale=10, min_width=0):
            clear_chat_button = gr.ClearButton(value="🗑️ Clear Chat",
                                               elem_id="clear_chat_button")
        with gr.Column(scale=10, min_width=0):
            summarize = gr.Button(value="📄️ Summarize Chat",
                                  elem_id="summarize_chat_button",
                                  variant="primary")
    clear_chat_button.click(clear_chat, inputs=[support_chatbot], outputs=[support_chatbot])
    audio_microphone.change(recognize, inputs=[support_chatbot, user_or_support_radio, audio_microphone],
                            outputs=[support_chatbot, audio_microphone, text_output])
    text_output.change(sentiment_analysis, inputs=[text_output], outputs=[])

app.queue()
if __name__ == "__main__":
    app.launch(server_port=8080, inbrowser=False, show_api=False)

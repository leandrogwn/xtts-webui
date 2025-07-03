import argparse
import os
import sys
import tempfile
from pathlib import Path

import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list,find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig  # Importa ambas clases
from TTS.config.shared_configs import BaseDatasetConfig  # Importa la clase faltante
from TTS.tts.models.xtts import Xtts

import requests

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {destination}")
        return destination
    except Exception as e:
        print(f"Failed to download the file: {e}")
        return None

# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def create_zip(folder_path, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
    return zip_path

def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        return create_zip(ready_folder, "optimized_model")
    return None

def get_dataset_zip(out_path):
    dataset_folder = os.path.join(out_path, "dataset")
    if os.path.exists(dataset_folder):
        return create_zip(dataset_folder, "dataset")
    return None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    
    # ⚠️ Agrega BaseDatasetConfig a las clases permitidas
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])
    
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the XTTS paths!"
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    
    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        speaker_file_path=xtts_speaker,
        use_deepspeed=False
    )
    
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    return "Model Loaded!"

def run_tts0(selected_language, lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None
#
    selected_speaker = speaker_audio_file
    selec_languaje = load_text_langs(selected_language)

    # Construct the file path
    speaker_audio_path = f"/tmp/Voice/{selec_languaje}/{selected_speaker}.mp3"

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_path, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_path

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


# Diccionario de idiomas y sus códigos
leng_and_ids = {
    "Select language": "es",
    "Voices Legacy": "show_legacy",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English-1": "en1",
    "English-2": "en2",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Spanish": "es",
    "Swedish": "sv",
    "Tamil": "ta",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi"
}

# Listas de nombres para cada idioma (Solución 2: Recomendada)
show_legacy  = ['Adam', 'Alice', 'Antoni', 'Aria', 'Arnold', 'Bill', 'Brian', 'Callum', 'Charlie', 'Charlotte', 'Chris', 'Clyde', 'Daniel', 'Dave', 'David_Martin._1', 'Domi', 'Dorothy', 'Drew', 'Elli', 'Emily', 'Eric', 'Ethan', 'Fin', 'Freya', 'George', 'Gigi', 'Giovanni', 'Glinda', 'Grace', 'Harry', 'James', 'Jeremy', 'Jessica', 'Jessie', 'Joseph', 'Josh', 'Laura', 'Liam', 'Lily', 'Matilda', 'Michael', 'Mimi', 'Nicole', 'Patrick', 'Paul', 'Rachel', 'River', 'Roger', 'Sam', 'Sarah', 'Serena', 'Thomas', 'Will']

arabic_names = ['Amr', 'Anas', 'HMIDA', 'Hamid', 'Haytham', 'Haytham_-_Conversation', 'Jafar_-_Deep_Narrator', 'Mo_Wiseman', 'Mona', 'Mourad_Sami', 'Raed', 'Sana', 'Wahab_Arabic']
bulgarian_names = ['Elena', 'Julian']
chinese_names = ['Coco_Li', 'Karo_Yang', 'Liang', 'Martin_Li', 'Maya_-_Young__Calm', 'ShanShan_-_Young_Energetic_Female', 'Stacy_-_Sweet_and_Cute_Chinese', 'YT']
croatian_names = ['Ivan', 'Luka_-_Narration', 'Maja', 'Slobodan']
czech_names_names = ['Anet', 'Hana_-_CZ', 'Hanka_beta', 'Jan', 'Jan_-_kind__gentle', 'Jiri', 'Ondřej_–_vypravěč', 'Pawel_TV™️_-_High_Quality_', 'Petr_Sovadina', 'Tony']
danish_names = ['Christian_-_Danish_calm_voice', 'Constantin_Birkedal', 'Mathias_-_Storyteller', 'Peter_-___Readings__Presentations', 'Sissel', 'Thomas_Hansen']

dutch_names = ['Arno_Drost', 'Bart', 'Daniel_van_der_Meer_', 'Jaimie_from_the_Netherlands_-_Dutch_Amsterdam_Voiceover_-_Young_Male_Age_30_', 'Richard', 'Serge_de_Beer_Pro1', 'Tijs']

finnish_names = ['Christoffer_Satu']

french_names = ['Adina_-_French_teenager', 'Adrien_Piret', 'Alexandre_Boutin_-_French_Canadian', 'Audiobooks_Lady', 'Audrey', 'Camille_Martin', 'Christophe_Géradon_Belge', 'Christophe_M', 'Claire', 'Coco_-_French_-_for_E-learning_and_Tutorial', 'Corentin', 'Cyril_-_Narration__Audiobook', 'Darine_-_Narration', 'Dave_-_Pro_Narrative', 'David', 'Denis_Landrieu', 'Emilie_Lacroix', 'Eric', 'Franck_de_France', 'Frédéric_-__French_Narration', 'Gaétan_L-Pro_French_Warm_Calm_Clear_Voice_Reader_conditions', 'Guillaume_-_French_voice_-_Narration_and_Voiceover', 'Guillaume_-_Narration', 'Haseeb_-_Canadian_French', 'Hélène', 'JaySoft', 'Jean_Petit_-_jeune', 'Jeanne_-_Professional_and_captivating_voice', 'Kevin_histoire_V2', 'Laurence_-_Class__Mature', 'Liam_-_Sharp__Pro', 'Louis_Boutin', 'Lucie', 'Lucien', 'Ludovic', 'Léo_-_Quebec_French', 'Léo_Latti', 'Mademoiselle_French_-_For_Conversational', 'Mademoiselle_French_-_for_Institutional_Video', 'Manuel_Formateur_-_Français', 'Martin_Dupont_Aimable', 'Martin_Dupont_Intime', 'Martin_Dupont_Profond', 'Mat', 'Mathieu_-_French_voice_-_Narration', 'Maxime_-_French_Young_Male', 'Maxime_Lavaud_-_French_young_man', 'Maxime_Lavaud_-_French_young_man_', 'Michel', 'Miss_French_-_For_Audiobook', 'Miss_French_Papote', 'Miss_Radio', 'Nicolas_-_Narration', 'Nicolas_Petit', 'Nicolas_Petit_-_Deep_voice_narration_', 'Nicolas_animateur_', 'Olivier_Calm', 'Patrick_-_Québec_Canada', 'Peter_-_Engaging_friendly_young_adult_male_voice', 'Romain_-_Lecture', 'Sam_French', 'SkaraB', 'Sophie_-_Pro_Audiobook', 'Sébastien_-_French_Male', 'Theo_-_Smart_warm_open', 'Ulys_-_Young__Energetic', 'Vincent_FR', 'Voix_Nicolas_Petit_ton_Animateur_Radio', 'Voix_grand_père']

german_names = ['Aaron', 'Albert_-_Funny_Cartoon_Character', 'Aleks', 'Alessandro_Devigus', 'Alex_-__Professional_German_Male_Voiceover', 'Amadeus', 'Ana', 'Ana_-_Novel_Audiobook', 'Andi_Brewi_-_Moderator_advertising_spokesperson', 'Andreas_-_Clear_German', 'Andreas_-_Deep_German_Voice', 'Annika', 'Anton_Dark_Magic_-_Thriller_-_True_Crime', 'Antonia_Konstanz_-_German_Native', 'Apollo_-_Documentary__TV_Voice', 'Ava_-_youthful_and_expressive_German_female_voice', 'Bartholomeus_Bösewicht_-_Grim_and_Gruesome', 'Ben', 'Ben_Hoffmann_-_German_Ads__Trailers', 'Carlos_-_der_Spanier', 'Carola_Ferstl_Nachrichten', 'Christian', 'Christian_Ehler', 'Christian_Kinderbuch', 'Clemens_Hartmann_-_The_Berlin_Voice', 'Clemens_Hartmann_2_-_for_Ads__Trailers', 'Clemens_Hartmann_3_-_The_Narrator', 'Cornelia_', 'Daniel_DaFraVe', 'Daniel_DaFraVe_Whisper._ASMR._Meditation._Relaxing', 'David_-_Serious_voice_for_narration_and_stories', 'Der_Beamte', 'Dimawalker_', 'Dimi', 'Dirk', 'Elias_-_Radio_Host__Radio_News_Presenter_Voice', 'Elias_-_Social_Media_Podcasts_Conversations__Discussions', 'Emilia_-_German_narrator', 'Fabian', 'Felix_-_Smooth_German_Chaos', 'Felix_-_Soft_Deep_German_Narration_Voice', 'Felix_Gebhardt_-_authentisch_und_berührend_Podcast_Hörbuch_Radio', 'Finnegan_Fairytale_-_Exciting_Childrens_Stories', 'Flauschi', 'Frederick', 'Frederick_-_Calm_Meditation_Deutsch', 'Frederick_-_Calm_and_Soothing_Meditation', 'Frederick_-_Friendly__helpful_', 'Frederick_-_Old_Gnarly_Narrator', 'German_Daniel', 'German_Michael_-_Loud_Clear__Striking', 'German_Voice', 'Grandpa_Georg_-_Funny_and_Gruff', 'Günther_Goodnight_-_Relaxed_and_Slow', 'Hans_Kraft', 'Heidi_factual_Standard_German_-_with_Swiss_Accent', 'Helmut_Schwarz', 'Herr_Gruber', 'Horvath_aus_Wien', 'Isabell', 'Jan', 'Jean_Art', 'Jesper', 'Johannes_-_Documentary_film', 'Jonas', 'Juan_Schubert', 'Julia', 'Julian_-_German_Explainer_Voice', 'Julius', 'Kris_Klingenberg', 'Kurt_-_Calm', 'Lana_Weiss_-_Meditation', 'Lea', 'Lena_-_Cute_German_Voice', 'Leo_liest', 'Leo_liest_tief', 'Leon_Stern_-_Fiction__Fantasy_', 'Leonie', 'Lex_Mystery', 'Lorenz', 'Louisa_', 'Luisa', 'Lukas_Harmony', 'Manuel_-_Your_Narrator_and_Storyteller', 'Marc', 'Marc_Weber_-_Non-fiction_books_', 'Marcel__Male__Audiobook__Tutorial__Trainings_GERMAN', 'Marco_-_Gentle_German_ASMR_Narrator', 'Marcus_KvE_–_German_Voice_Over', 'Marie_-_German_Frenchwoman', 'Marko_-_German_Male_Deep_Voice', 'Markus', 'Martin_History', 'Martin_Jung', 'Martin_R._Pro', 'Max_Mustermann_-_Ernst', 'Meine_Lesestimme', 'Michael', 'Mila', 'Nader', 'Narrator_Markus', 'Niander_Wallace_', 'Otto', 'Patrick_-_German_speaker', 'Peter_Hartlapp_-_Voiceactor_Werbesprecher_und_Moderator', 'Peter_Meta_Business_Twin', 'Petra_PeFraVe_Pro_', 'Petra_PeFraVe__-_Funny', 'Phil_-_Fantasy__Thriller', 'Philipp_-_Male_with_standard_accent', 'Prinz_Pricklig_-_Whispering_Sparkling_and_Crisp_', 'Rafi_Biber', 'Reeloverlay', 'Rob', 'Robby_-_Audio_books_Speeches__Stories', 'Robert_dein_freundlicher_Assistent', 'Robert_erklaert_mit_Betonung', 'Robert_hypnotisiert_entspannte_Meditation', 'Samer', 'Sammy_Zimmermanns', 'Sascha_Pro_', 'Sebastian_Thomas', 'Stefan_Rank_der_Erzähler_Radio-Moderator', 'Susi', 'Sympathische_Stimme', 'Thomas_-_The_pragmatist', 'Timo', 'Tom_-_Deep_German_Voice', 'Tom_Magic', 'Tommy_Studio_Voice_2', 'Torsten_-_Raspy_Charmer', 'Tristan_Medersburg_-_Trustworthy_Deepness', 'Vali_-_Young_man_with_a_bass-heavy_voice', 'Vincent_-_Factual', 'Willi_-_Professional_German_Narrator']

greek_names = ['Agapi', 'Fatsis_', 'Giassiranis_Dimitrios', 'Kyriakos', 'Niki_-_native_Greek_female_', 'Niki_2_-_native_Greek_female', 'Niki_3_-_native_Greek_female', 'Stefanos_-_Calm_youthful_and_casual', 'Takis_-_native_Greek_male']

hindi_names = ['Aaditya_Kapur_-_Calm_Conversational_Hindi_Voice', 'Aakash_Aryan_-_Conversational_Voice', 'Amit_Gupta', 'Anand_-_Storytelling_and_Narration_Hindi', 'Anoop', 'Ayesha_-_Energetic_Hindi_Voice', 'Bobby_', 'Danish_Khan_-_Expressive_Old_Voice', 'Devi_-_Clear_Hindi_pronunciation', 'Faiq_-_Standard_Hindi', 'God', 'Guru_-_Rich_Bass_Hindi_Voice', 'Ishika_Singh_-__Storytelling_and_Narration_Hindi', 'Janvi_-_Expressive_Indian_Voice_', 'Jitu', 'John_-_Confident_and_Deep', 'Kaaya_-_Gentle_Hindi_', 'Kanika_-_Relatable_Hindi_Voice', 'Krishna_-_Energetic_Hindi_Voice', 'Kunal_Agarwal', 'Leo_-_Energetic_Hindi_Voice_', 'Luv_-_Hindi_Storytelling_Voice', 'Manu_-_Smooth_Modulated_Voice', 'Monika_Sogam_-_Hindi_Modulated', 'Muskaan_-_Casual_Hindi_Voice', 'Natasha_-_Energetic_Hindi_Voice', 'Neel_-_Expressive_Narrator', 'Nikita_-_Youthful_Hindi_Voice', 'Nipunn_-_Deep_Hindi_voice', 'Niraj_-_Hindi_Narrator', 'P_K_Anil_-_Clear_Hindi', 'Parmeshwar_परमेश्वर', 'Parveen_-_Hindi', 'Pratima_-_Casual_Hindi_Conversational_Voice', 'Prem_-_Connectable_Hindi_Voice', 'Priya', 'Raju_-_Relatable_Hindi_Voice', 'Ranbir_Merchant_-_Deep_Engaging_Hindi_Voice', 'Ranga_-_Authoritative_and_Deep_Hindi_Voice', 'Reva_-_Familiar_Hindi_Voice', 'Riya_K._Rao_-_Hindi_Conversational_Voice', 'Ruhaan_-_Clean_Hindi_Narration_Voice', 'Saanu_-_Soft_and_Calm', 'Sachin_-_Deep_and_thoughtful', 'Saira_-_Young_Casual_Voice', 'Samads_Realistic_Voice', 'Shakuntala_-_Expressive_Indian_Voice', 'Shrey_-_Deep_Hindi_Voice', 'Sohaib_Jasra_', 'Sonu_Indian_Male', 'Suhaan_-_Delhi_Guy', 'Sweetie', 'Vihan_Ahuja_-_Friendly_Hindi_Voice', 'Yash_A_Malhotra_-_Warm__Friendly_Hindi_Voice', 'Zadok_-_Good_for_character']

hungarian_names = ['Magyar_Férfi_-_Hungarian_Male', 'Susanna_Rutkai']

indonesian_names = ['Abyasa', 'Andi', 'Andra', 'Bambang__', 'Bee_Ard_-_Clear_Dynamic_Voice', 'Blasto', 'Hendro_Atmoko', 'Jin', 'Mahaputra', 'Meraki_female_Indonesian_voice', 'Miz', 'Pramoedya_Chandra', 'Pratama', 'Putra', 'Suara_narasi', 'Tri_Nugraha_Ramadhani', 'Zephlyn']

italian_names = ['Aaron', 'Alessandro', 'Alessio_-_positive_and_professional', 'Andrea_Loco', 'Anna', 'Antonio_Farina_-_Italian_PRO_Talent_-_Audiobook_Narration', 'Carmelo_La_Rosa_-_Italian_Pro_Talent_e-learning_news_webinar_istitutional.', 'Chris_Basetta_-_Audio_Books', 'Chris_Basetta_-_Social_Media', 'Dante_-_Italian_30_years_old', 'Emanuel', 'Eray_Rio·Sae', 'Fabi', 'Francesco', 'Francesco_-_Narrative', 'Francesco_-_Premium', 'Gabriele', 'Germano_Carella', 'GianP_-_Edu_-_Clear__Upbeat', 'GianP_-_Narrative_Storytelling', 'GianP_-_News_Info_and_Documentary', 'GianP_-_Social_Media__Ads', 'Gianluigi_Toso', 'Giovanni_Rossi_-_giovane', 'Giulia_-_sweet_and_soothing', 'Gus_-_Deep_and_Pleasant', 'Kina_-_Cute_happy_girl', 'Leandro_', 'Linda_Fiore', 'Luca', 'Luca_Brasi_Gentile', 'Luca_Brasi_Intimo', 'Luca_Brasi_Profondo', 'Luna', 'Marcello_Lares_-_Soothing_Narrator', 'Marco', 'MarcoTrox_-_Italian_Pro_Voice_Actor_-_Storytelling_Audiobooks_Narration.', 'MarcoTrox_-_Italian_Professional_Voice_Talent', 'Marco_Pro', 'MrVibes', 'Nicola_Lorusso_-_Italian_Pro_-_Storytelling_Audiobooks_Narration.', 'Oceano_-_A_very_young_narrator', 'Pietro_-_Crazy_Character_Narrator', 'RenzoTech_', 'Stefano', 'Stefano_Becciolini_1']

japanese_names = ['Asahi_-_Japanese_male', 'Ena_', 'Hinata', 'Hiro_Satake', 'Ichiro', 'Ishibashi_-_Strong_Japanese_Male_Voice', 'Junichi', 'Ken', 'Ken_-_Japanese_male', 'Kozy_Male_Japanese_Narrative_Voice_-_Tokyo_Standard_Accent', 'Morioki', 'Otani', 'Sakura_Suzuki', 'Shoki']

korean_names = ['Anna_Kim', 'Bin', 'ChulSu', 'Do_Hyeon', 'Funny_Jackie_Lee', 'HYUK_', 'Hyuk', 'Hyun_Bin', 'Jaedong_Ahn', 'Jina', 'Jung_-_Narrative', 'KKC', 'KKC_-_Guided_Meditation__Narration', 'Kyungduk_Ko', 'Man_Bo', 'Min_ho']

norwegian_names = ['Johannes_-_Norwegian_-_Upbeat', 'Mia_Starset']

polish_names = ['Adam_-_Polish_narrator', 'Adygeusz', 'Aneta_-_Loud_and_confident_voice', 'Ave_Cezar', 'Bart', 'Bea', 'Damian_PL_', 'Daniel', 'Dawid_PL', 'Ignacius', 'James_-_Narrative__Story', 'Jerzy', 'Krzysztof_PL', 'Lena_Suzuki', 'Maciej', 'Maciek', 'Mark_-_Polish', 'Martin', 'MePolish', 'Mr_Lucas_', 'Oliver_Brown', 'Pawel_Pro_-_Polish', 'Piotrek_Pro', 'Pixi', 'Robert', 'Robert_Rob']

portuguese_names = ['Adriano_-_Narrador3', 'Adriano_-_Narrator', 'Adriano_-_Narrator2', 'Alcione', 'Ale_Garcia', 'Ana_-_Brazilian', 'Ana_Dias', 'Andreia_I.', 'Bia_-_Brazilian', 'Brazilian_Dudy', 'Conrado_Bueno', 'Daiane_Candido', 'Daniel_Dan', 'Davi', 'Dhyogo_Azevedo', 'Diego', 'Eddie_Barroso_-_Brazilian', 'Edna_E.', 'FMDAmbrosio', 'FRANCISCO_IA', 'Fabio_Filho', 'Flavio_Francisco_-_Narrative_-_Brazilian_Portuguese', 'Gabby', 'Gilson_Lima', 'Gustavo_Barros', 'Gustavo_Jannuzzi_', 'Gustavo_Sancho', 'Higor_Bourges', 'Hugo_Mendonça', 'João_Pedro', 'Juliana_Barbieri', 'Keren_-_Young_Brazilian_Female', 'Klaus_-__Young_Brazilian_Professional_Narrator', 'Kuhcsal', 'Lax', 'Leonardo_Hamaral', 'Locução_para_Propaganda', 'Luka', 'Marcelo_Costa_Brasileiro', 'Matheus_-_Energic_Young_Voice', 'Michele_-_Brazilian', 'Muhammad_Umm', 'Oliveir4_Music', 'Onildo_F._Rocha', 'Otto_de_La_Luna', 'Papai_Noel_', 'Rafael_Valente_-_Brazilian_Professional_Narrator', 'Rener', 'Roberto_Barbieri', 'Rodrigo_Rodrigues', 'Samuel_-_Jovem_Empreendedor', 'ScheilaSMTy', 'Slany', 'Thiago_Realista', 'Vagner_De_souza', 'Vinicius_Bergamo', 'Wesley_Bessa_', 'Weverton_', 'Will_-_Deep']

romanian_names = ['Andrei', 'Antonia', 'Apeiron', 'Ciprian_Pop', 'Corina_Ioana', 'Cristi_Romana', 'Cristina_Amza', 'Jora_Slobod', 'Liviu_Mihai', '_Bogdan_-_Advertising']

russian_names = ['Aleksandr_Petrov', 'Andrei_-_Calm_and_Friendly', 'Anna_-_Calm_and_pleasant_', 'Artem_K', 'Artemii_Levkoy', 'Dimitri', 'Dmitry', 'Felix_-_calm_friendly', 'Larisa_Actrisa', 'Marat', 'Mark_Rozenberg', 'Max_-_Clear__Professional', 'Nadia', 'Nikolay', 'Oleg_Krugliak_', 'Oleksandr_Trotsenko', 'Ranger3D.pro', 'Tyler_Soapen', 'Viktoriia_-_clear_resonant_young_female_voice']

slovak_names = ['Andrej']

spanish_names = ['AF', 'Alberto_Rodriguez', 'Alejandro_-_Mexican_male', 'Alejandro_Aragon', 'Alejandro_Ballesteros', 'Alejandro_Durán', 'Alex_-_Happy_Upbeat_Joyful_Energetic', 'Alex_Comunicando', 'Andrea', 'Andrew_V.', 'Andromeda_Thunders', 'Angie_vendedora_Colombiana', 'Ani_Egea', 'Ani_Egea_-_Expressive', 'Antonio_LV', 'Antonio_ia', 'Apex_-_Fitness_-_Nutrition_-_Coach_-_Energetic_-_Professional', 'ArthisRap_Pro', 'Ashley_Travels-_American_English_Tourist_speaking_Spanish_', 'Bardo_Limon_-_Epic_Promotional_Voice', 'Bebe_Lunita_-_Bebe_hablando', 'Beto_-_Latin_American_Spanish_Argentina', 'Bruno_-_Suspense_-_Thrill_-_Horror_-_Tense', 'Brêchet_Simon', 'CRISTINA_VOICE', 'Carles_Pujol', 'Carlos_-_Podcasting__News', 'Carmelo', 'Carmelo_Crespo', 'Carmelo_Crespo_-_Expressive', 'Carolina_-_Spanish_woman_-_es_ES', 'Christian_Avilés_-_documentales_e-learning_corporativos_y_Redes_Sociales', 'Claudia_Whispers-_Asmr_Spanish_Intimate', 'Cristi_Poot', 'Cristian_Medina', 'Damian_Valdez', 'Dan_Dan', 'Dante_-_Castilian_Spanish', 'Dany_-_Professional_narrator', 'David_Martin._1', 'David_Martin_2', 'Denilson', 'Didak_Leñero__Spanish_Spain', 'Diego_Aguado_-_Spanish_deep_voice', 'Diego_Cárdenas', 'Diego_Galán', 'Dominican_', 'Dosi_Español', 'EDGARD', 'Eduardo_-_Advertising__Commercial_voice_in_Spanish', 'Eduardo_M._-_Mexican_Spanish', 'Eduardo_Román', 'Efrayn', 'Eleguar_-_Latin_American_Spanish', 'Eleguar_-__Deep_Latin_American_Spanish', 'Emiliano_Zamora', 'Emilio_Menal', 'Enrico', 'Enrique_M._Nieto', 'Enrique_Mondragón', 'Erika_-_Raspy_and_Pleasant', 'Eva_Dorado', 'FantasyCraft_Studios', 'Fer', 'Fernanda_olea_1', 'Fernando', 'Fernando_Martinez', 'Firusho', 'Francisco', 'Frankie_San_Juan', 'Gabriela_-_Spanish_from_Mexico_', 'Gabriela_Gonzalez_', 'Gilfoy', 'Ginyin', 'Ginyin_2_-_Webpages_Narrative__Books', 'Grandma_Titina_-_70_year_old_woman', 'Guillermo_Brazález', 'Guillermo_Brazález_-_Dynamic__Cheerful', 'Haroldo_', 'Hernán_Cortés', 'Isabela_-_Spanish_Childrens_Book_Narrator', 'Jacson_Ander', 'Jaime_Fregoso_-_Professional_Annoucer', 'Jaime_Tu_Locutor_Online', 'Jarpa_Test_-_Francisco', 'Jav_-_Calm_clean_and_profound_voice', 'Javier_España', 'Javier_Madrid', 'Javisanchez', 'JeiJo_', 'Jhenny_-_Warm_Fluid_and_Smooth', 'Jhenny_Antiques_-_Calm_Soft_and_Sweet', 'Jonathan', 'Jorge', 'Jorge_Gaviria_-_Powerful_and_impactful', 'Jorge_Mario_-_Spanish_to_read_books_and_narration', 'Jose_A._del_Rio', 'José_Borda', 'José_Borda_-_Deep', 'José_Borda_-_Expressive', 'Juan', 'Juan_Carlos', 'Juan_Manuel', 'Juan_Manuel_-_Conversational', 'Juan_Pablo', 'Kiko_Hdz', 'Knight_JAVIER-Calm_Gentle', 'Lalo', 'Leo_-_Energetic_Warm_Happy_Upbeat_Inviting_Optimistic', 'Leo_Kid_Spanish-_Character', 'Leonardo', 'Ligia_Elena', 'Ligia_Mendez', 'LoidaBurgos', 'Luis', 'Luis_Guary', 'Luis_R_Casiano', 'Luis_Vega', 'Lumina_-_Clara__Natural', 'Maicolangel', 'Malena_Tango', 'Mariluz_Parras', 'Mariluz_Parras_-_Expressive', 'Martin_Osborne_1', 'Martin_Osborne_2', 'Martin_Osborne_4', 'Martin_Osborne_5', 'Martin_Osborne_6', 'Martin_Osborne_7_', 'Mary', 'María', 'Mauricio', 'Mauro_C', 'Maxi_Araya', 'Maxi_Argames', 'Memo_M_-Professional_Latin_American_Spanish', 'Mia_García-_business_narrations_and_informative', 'Mia_Instructor-_Spanish_E_learning_corporate_Conversational_training', 'Miguel', 'Mikel_-_Adulto_idioma_español', 'Miquel', 'Nina', 'Oliver_Podcasting_Refinada', 'OmarVoice', 'Omgpvoice', 'Omgpvoice_-_Expressive', 'Pablo_Vambe_AI_V2', 'Paloma_S.__-_Spanish_-_Conversational_Comforting_Compelling', 'Pilar_Corral', 'Rafael', 'Regina_Martin', 'Ricardo', 'Rodolfo_Rodriguez_', 'Rosa_-_Spanish_Calm_Old_Woman', 'Rosa_Zambrano_', 'Santiago', 'Santiago_-_calm', 'Sara_Martin_1', 'Sara_Martin_2', 'Sara_Martin_3', 'Screaming_George', 'Serena_AI', 'Sergio_Juvenal', 'Sofi', 'Soy_Luis_Cen', 'Tatiana_Martin', 'Tony_Villa', 'Valeria', 'Victor', 'Víctor_Hinojosa', 'Yinet_-Upbeat_Columbian_Woman', 'Yorman_Andres', 'Zabra_-_Commercial_Announcer', '_Medellin_-_Colombian_Voice', 'paco']

swedish_names = ['Adam_Composer_Stockholm', 'Jonas_calm__informative_Swedish_voice', 'Sanna_Hartfield_-_Sassy_Swedish_', 'Sanna_Hartfield_-_Swedish_Conversational', 'Sanna_Hartfield_-_Swedish_Narration']

tamil_names = ['Ashwin_-_Relatable_Tamil_Voice', 'Madsri_-_Friendly_Tamil_Voice', 'Madsri_-_Tamil_Narrator', 'Meera_-_Conversational_Tamil_Voice', 'Nila_-_Warm__Expressive_Tamil_Voice', 'Ramaa_–_Energetic_Conversational_Tamil', 'Ramaa_–_Energetic_Tamil_Narrator']

turkish_names = ['Adilcan_Demirel', 'Ahmed', 'Ahmet_Evlice', 'Ahmet_Çiçek', 'Arman_Yılmazkurt', 'Belma_-_Dynamic_Playful_Clear_Narrator', 'Burak_Yoglu', 'Burcu_Basyigit', 'Cagatay_A.', 'Calm_Turkish_AudioGuide', 'Cavit_Pancar_-_Epic_Powerful_Historical', 'Cem', 'Cicek_-_Joyful_Dynamic_Storyteller', 'Derin_Roman_-_Epic_Dark_Powerful', 'Doacast_', 'Doga', 'Eda_Atlas', 'Emre', 'Emre_Gökçe', 'Farshid', 'Fatih', 'Fatih_Çetinkaya', 'Furkan_Keser', 'Gokce', 'Gokce_lx', 'Gozde_Arikan', 'Gönül_Filiz', 'Hakan_Turk', 'Halil_', 'Hulya', 'Hurrem_-_Confident_Turkish_Actress', 'Ipek_-_Professional_Confident_Narrator', 'Irem', 'Kamil', 'MUHAMMER_ARABACI', 'Mad_Scientist_-_For_All_Languages', 'Mahidevran_-_Playful_Clear_Powerful_Narrator', 'Mert', 'Mertkan_Erkan', 'Mustafa_Can', 'Onur_Can', 'Onur_Naci_Ozturkler_-_spunkram', 'Ramazan', 'Recep_Arkiş_', 'Rıdvan_Elitez', 'Se_-_Young_Male_Reading', 'Sedat', 'Sencer', 'Seyda_-_Eğlenceli_Anlaşılabilir_Fun_Fluent_Clear', 'Sohbet_Adami_-_Natural_Chat_Friend', 'Sultan_-_Charming_Seductive_Narrator', 'Tarik', 'Tuba_Velidede', 'Tuncay_Saran', 'Valperga', 'Walter_BJ', 'Whispering_Irem', 'Yigit', 'Zafer_', 'bilgehan', 'İbrahim_Halil_Acioglu', 'İbrahim_Khan_İpek']

ukrainian_names = ['Anton', 'Danylo_Fedirko', 'Dmytro_UA', 'Oleksii_Safin', 'Olena', 'Volodymyr_Pro']

vietnamese_names = ['Actor_Pham_Hung', 'Announcer_Van_Phuc', 'Ca_Dao', 'Kim_Tuyến', 'Ly_Hai', 'MC_Duy_Minh', 'Mai', 'Nhung', 'Sơn', 'Trang', 'Trung_Caha', 'Tuan_TLU']

english1_names = ['2B_Impression', 'ANDREA_CUTE_female_voice', 'Aakash_Aryan_-_Conversational_English_Voice', 'Aaron_-_Monotone_tech_narrator', 'Aaron_-_trusted_and_engaged', 'Abandoned_school', 'Abigail_-_arrogant_and_snobbish', 'Abrogail_', 'Ada', 'Adam__-_Newscaster', 'Adina_-_Teen_Girl', 'Aditi', 'Adriano_-_44', 'Aerylla', 'Aiden_-_Happy_Video_Host', 'Ailema_-_calm__Soft', 'Akwasi_-_Young_Ghanaian_man', 'Al', 'Alan', 'Alec_-_Energetic_Confident_and_Charismatic', 'Alex', 'Alex_-_Vibrant_Engaging_and_Lively', 'Alex_the_Performer_-_Commercial_Warm_Inviting_Expressive', 'Alexander_-_Mature_and_confident', 'Alexi', 'Alexite', 'Ali_', 'Alice_-_calm_and_soft_narrator', 'Alisha_-_Soft_and_Engaging', 'Alton', 'Alyx_-_Vibrant_British_Male', 'Amada', 'Amanda_-_a_natural_narrator', 'Amar', 'Amelia', 'Amelia_-_young_and_soft', 'Amilia', 'Amina_-_regal', 'Amritanshu_Professional_voice', 'Amy_-_Clear_and_Concise', 'Amy_-_Smart_Teacher_Narration', 'Amy_-_Witty_College_Girl', 'Andre_LeDoux_-_Romantic_Fancy_Talking_male_', 'Andrew', 'Andrew_-_Old_slow_voice', 'Andrew_-_Smooth_audio_books', 'Andrew_Radio', 'Andy_Berg', 'Angie_-_Upbeat_Book_Narrator_Professional_Videos_Engaging_Conversations_Radio_News_Meditation', 'Anjali', 'Anjina', 'Anna_-_Modern', 'Annie', 'Anthony_-_emotive__expressive', 'Arabella', 'Arayah_-_Mature_and_Professional', 'Archer', 'Aria_-_Sexy_Female_Villain_Voice', 'Armando', 'Armando_realistic', 'Asarte', 'Ash_', 'Asher_Avery_Alex_-_Engaging_and_Real__Storyteller_and_Performer', 'Asmodia_-_earnest', 'Aspexia_-_Grand__Clear', 'Athena_-_Stern_serious_and_powerful', 'Attention_Grabbing_Male_Narrator', 'Aunt_Annie_-_calm_and_professional', 'Aurelia_-_High_Quality_Realistic_Princess_', 'Aurion_-_Wise_Narrator', 'Austin_-_Dramatic_Narration', 'Austin_Boy', 'Austin_the_Cowboy', 'Ava', 'Ayden', 'Ayesha_-_Energetic_Indian_Voice', 'Ayinde_-_young_British_Nigerian', 'Bailey_-_twenty-something_earnest_confident', 'Bails', 'Barry_Bob_Alone', 'Bateman_-_Deep_Masculine_and_Authoritative', 'Befutig_-_Steady_Robust__Engaging', 'Befutig_Safiza_Uj-alet_-_Resonant_Commanding__Authentic', 'Belinda_-_Curious_and_Soft', 'Bella-_sensual_allurin_beautiful', 'Bella_-_Direct_and_Understanding', 'Belle_-_Clear_Well-Modulated_Expressive', 'Ben', 'Ben_-_British_male_young', 'Benjamin_-_The_Frenchy_Guy', 'Benjamin_S_Powell', 'Benny', 'Bert', 'Beth_-_gentle_and_nurturing', 'Betsy_-_Wise_and_Thoughtful', 'Betty', 'Beyond_Average_Joe', 'Bhavna_-_Insightful_Storyteller', 'Bianca_-_City_girl_', 'Bill_-__A_deep_voice_narrator', 'Bill_Oxley_-_Clear_informative_mature_forthright_and_understandable', 'Blaire_Frost', 'Blkking407_', 'Bob_-_old_man', 'Bogdan_-_Soft_Male_Narration', 'Boi', 'Booney_-_calm_and_cute', 'Brandon_-_Young_Male_American_Voice_Over', 'Brandon_Cole', 'Brandon_VO_Artist_Clone', 'Brayden_-_Conversational_Older_Teen', 'Brenda', 'Brenda_-_Raspy_female_', 'Bria_-_Young_and_Soft', 'Brittney_-_Male_Child_-_Youthful_Raspy_Cute__Excitable_', 'Brittney_-_Social_Media_Voice_-_Fun_Youthful__Informative', 'Brody_-_Serious', 'Broom', 'Bruce_-_vibrant_and_baritone', 'Bryan', 'Bryn_-_Calm_and_Expressive', 'Bud', 'Cal_-_confident_professor', 'Calliope_-_ancient_muse', 'Camelia', 'Cara_-_Expressive_and_Direct', 'Carl_-_Big_Voice', 'Caroline_-_clear_and_confident', 'Carter_-_Caring_and_Rational_British_Male', 'Cassandra', 'Cassandra_-_Confident_and_Vibrant', 'Cassia', 'Catherine_-_Professional_and_Direct', 'Cecile_-_Confident_and_Strict', 'Charlie_-_Posh_and_Royal', 'Charlotte_-_precise', 'Chazza_Hypno', 'Chechi_for_first_video', 'Chelsea_Boddie', 'Chinmay_-_Calm_Energetic__Relatable_', 'Chloe_-_sharp', 'Chris_-_irritable_boss', 'Chris_C_-_Mid_30s_-_Podcast_Reviewer_good_for_shorts', 'Chris___Young_and_Inspired', 'Chrissy_-_Millenial_Female', 'Christian_Rivera', 'Christina_-_Trained_on_over_900_characters_with_emotional_dialogue', 'Christopher_', 'Christopher_-_friendly_guy_next_door', 'Chrisva', 'Ciro_-_real_intense_twentyish', 'Clara', 'Claw_Benn', 'Cody_-_Energetic_Upbeat_Educator', 'Cody_McAvoy', 'Cole_-_Gritty-Rough-Strong', 'Cooper', 'Cornelis', 'Creator', 'Cristiano', 'Crystal', 'Cyrus', 'Dan', 'Dana', 'Danbee', 'Daniel', 'Daniel_-_American_Game_Show_Host', 'Daniel_-_expressive_and_wise', 'Danny_-_highschool_jockish', 'Daphne_-_alluring_goth', 'Dara_-_loud_and_Intense_', 'Darwin_-_Rich_Mature_Voice', 'Daryl', 'Dath_Ilan', 'David', 'David_-_British_Storyteller', 'David_-_Deep_British', 'David_Bent', 'David_Castlemore_-_Newsreader_and_Educator', 'David_DeWitt', 'David_Eclipse', 'David_Esposito', 'David_Hertel', 'Dean_-_Goody_Two_Shoes', 'Deb_-_emotive_and_expressive', 'Deja', 'DellaRayne', 'DellaRayne_-_Smooth_and_Assertive', 'Demon_Monster', 'Derrick_-_melancholy', 'Desdemona_-_sassy', 'Desmond_-_clear_sincere_angst', 'Dezzy_-_Young_and_Soft', 'Dhyogo_azevedo', 'Diana_-_Meditative_Calm', 'Donny_-_Real_New_Yorker', 'Donny_-_very_deep', 'DrRenetta_Weaver', 'Dr_Lovejoy_-_Pro_Whisper_ASMR_', 'Drake__Warm_Canadian_English', 'Drew', 'Drew_-_Deep_Soothing_Guided_Meditation', 'Duke', 'Durgesh', 'Eamon_-_old_lecturer', 'Ed_Holderness', 'Edward', 'Egbert_-_upbeat_meditations', 'Elisa', 'Elisabeth_-_meditative', 'Elizabeth', 'Elizabeth_-_Wise_and_wistful', 'Elizabeth_-_calm_commanding_classic', 'Ella_-_Old_And_Deep_', 'Ellie', 'Emily_-_Australian_Female', 'Emily_-_relaxed_and_conversational', 'Emily_-_sweet', 'Emma', 'Emma_', 'Emma_-_A_brilliant_young_magician', 'Emma_-_sharp', 'Emma_Taylor', 'Emmeline_-_a_young_clear_and_confident', 'Epiktet_Philosoph_', 'Erdem_-_Educational_and_Instructional', 'Erika', 'Erin_-_Meditation_Guide', 'Ethan', 'Ethan_-_expressive_wise', 'Eustis_', 'Evan_-_showbiz_excited_happy', 'Eve_-_young_Australian_girl', 'Ezreal', 'Ezreal_-_energetic', 'Faith', 'Feeven', 'Female_Romance_Novel_', 'Foxy_-_Futuristic_Robotic_Personal_AGI', 'Francesca', 'Frank', 'Frank-_scary_stories', 'Frank_Johnson', 'Fucia_-_Youthful_and_Confident', 'Gabriella_-_deep', 'Garrett_Wasny', 'Gault_-_Youngish_excitable_high-strung.', 'General_Joe_-_WWII_Narrator', 'George', 'George_-_Serious_and_Experienced', 'Gerhard_Bakker', 'Gertrude_-_Childrens_Narrator', 'Gijs', 'Gladys', 'Goddess_Freyja_-_A_Mysterious__Magical_Muse', 'Graham_-_Old_and_Wise', 'Greg_Murphy', 'Gregoria', 'Gruhastha_-_Energetic_Enthusiastic__Articulate', 'Guy', 'Hakim_-_Audiobook_English__Arabic_Gulf_Accent', 'Halbert', 'Halley_McClure', 'Hallie_-_soft-spoken_and_subtle', 'Hallie_-_youthful_girl_voice', 'Hamlin_-_Deep_and_Booming', 'Hannah___Confident_Teacher', 'Hardcore_Henry_-_Intense_Storyteller', 'Harold', 'Harry_-_Proper_and_Academic', 'Harry___Sad_Emotional_Reck_', 'Harvey_-_Knowledgeful_Upfront', 'Haven_Sands', 'Helena_-_British_female_gentle_and_smart', 'Hemaka', 'Herbie_-_Lisp_and_whistle_S_sounds', 'Hermes_-_frank_abrupt_messenger', 'Hobbs_-_Casual_Narration', 'Horace_-_intense_deep_elder', 'Huckleberry_-_Southern_Charm', 'Hyde', 'Ian', 'Indian', 'Ingmar_-_Intimately_Mysterious', 'Investigator_Jane', 'Iomedae', 'Iris', 'Isabel_-_emotional__lisp', 'Isabella_', 'Isla_-_Strong_British_accent', 'Isla_Reid', 'Ivan_the_Mighty', 'Ivy_-_Free_Spirit', 'J._Thorn', 'J._Tyson', 'Jack', 'Jack-_Raspy__deep', 'Jacme', 'Jade', 'Jakobi_-_Emotive_and_Intriguing', 'James_-_Deep_', 'James_-_Deep_and_Booming', 'James_-_cool_and_expressive', 'James_-_deep_and_to_the_point', 'James_Fitzgerald', 'Jami_-_Mature_and_Clear', 'Jannice', 'Jason', 'Jason_Jordan', 'Jason_Pike', 'Jasper_-_androgynous_and_rebellious', 'Jasper_-_erudite_and_inquiring', 'Jeff', 'Jeff_-_Australian_Male', 'Jeff_-_Smooth_and_Confidant', 'Jennifer_-_expressive_and_cheerful_narrator', 'Jeremy_-_meditative', 'Jeremy_Clarkson', 'Jeremy_Smith', 'Jerry', 'Jerry_-_Energetic_and_Upbeat', 'Jessica_Anne_Bogart_-_Conversations', 'Jeż', 'Jim', 'JimBob_', 'Joan', 'Jodie_-_Assertive_and_Intelligent', 'Joe_-_American_Male_Narrator', 'Joe_-_professional_British_male_voiceover', 'Joe_02', 'Joey_Reeve', 'John', 'John_-_Deep', 'John_-_Guided_Meditation__Narration', 'John_-_Ultra_Brutal_Man', 'John_Beamer', 'John_Doe_Gentle', 'John_Domus_Cruo_-_Serious', 'John_Fernandes_-_Energetic__Friendly', 'Johnny_-_Upbeat_Professional_American_Male', 'Johnny_Boy_-_Action_Movie_Narrator', 'Johnny_Kid__-_Serious', 'Johnson_-_American_Male_voice_', 'Jones_-_Articulate_Gruff_Raspy', 'Jordan', 'Jordan_-_Warm_Narrator', 'Josh', 'Josh_-_Quiet_Person', 'Josh_T.', 'Joshua_-_Authoritative_Warm_and_Articulate', 'Joshua_-_Young_Soft_Warm_Male_Voice', 'Judy_-_Aged_and_Confident_Elder', 'Julie_-_expressive_and_energetic_romance_narrator', 'Justin_Time_-_eLearning_Narration', 'Justine_-_Expressive_Teen_Boy', 'Kade_Murdock_2.0', 'Kala', 'Kallen', 'Karen', 'Karma_-_Professional_and_Thoughtful', 'Kasi', 'Kathleen_Julie_-_alto_serious_articulate_focused_and_direct', 'Katy_-_sassy_teen', 'Kayla_-_Nurturing_and_Caring', 'Kelli-_Young_Mature_Southern', 'Kelly', 'Kelly_-_clear_teen_voice', 'Kenneth_-_strange_eccentric_old_gentleman', 'Kenny_-_Volume_2', 'Kevin', 'Khaled_', 'Khushi_-_New_Indian_Voice', 'Kieran_-_newsreader_male', 'Kik', 'Kim_-_Swedish_accent', 'Kim_Selch_-_Pro_Studio_Recording', 'King_Chuku', 'Kingsley_-_dapper_and_deep_narrator', 'Kirsten_-_Elegant_Knowledgeable_and_Reassuring', 'Kirt', 'Kitten_Kaley_Rose', 'Kiwi_-_Holistic_Educator', 'Kristopher_-_Gentle_ASMR_', 'Kuk', 'Kurrayah_-_young_and_friendly', 'Kuthon', 'Kwame', 'Kyana_Cook', 'LIAM_DALE', 'Lalitha_J_-_Tamil_Old_Woman', 'Lamar_Lincoln-_Black_Male', 'Latisha', 'Laurance', 'Lawrence_Mayles', 'Lee__Middle-Aged_Australian_Male', 'Leif_-_husky_male', 'Lena_-_crispt_and_confident', 'Leonardo_', 'Lerato', 'Liam', 'Liam_', 'Lila_-_Intelligent_and_emotive', 'Lily', 'Linus_-_A_young_American_tech_video_narrator', 'Lisa___Stern_and_Assertive_', 'LiveCat', 'LiveChi_', 'Liz', 'Lloyd', 'Lucas_-_motivational_speaker', 'Ludo_-_Storyteller_-_Your_epic_story_narrator', 'Luis_Gabriel', 'Lukas', 'Luna_-_Well_rounded_insightful_charismatic', 'Lyle', 'MANSHI', 'Magnolia_-_Mature_and_Wellspoken', 'Magpie', 'Marc_--_Smart_Soothing_Man', 'Marco-_hot_male_voice', 'Margot_', 'Mariam', 'Marianne_-_Narrative_Friendly_British_', 'Maribeth_-_A_Southern_Sweetheart_', 'Marie_KC', 'Marilyn_-_confident', 'Marissa_-_Friendly_and_Sociable', 'Marissa_from_ElevenLabs', 'Marjorie_', 'Mark', 'Mark_-_Very_Deep_Confident_Professional', 'Mark_-_Young_and_Calm_', 'Mark_-_clear_and_professional_newscaster', 'Mark_-_confident', 'Mark_-_raspy', 'Markus_-_Mature_and_Chill', 'Marshal_-_Dandy_Brit', 'Marshal_-_Toon_Character', 'Marta_-_Officious', 'Matt_Landon_', 'Matt_Rogo', 'Matt_Washer', 'Matthew_-_American_Male_Narrator', 'Matthew_-_Friendly_Clear_and_Perfect_for_Educational_Content', 'Matthew_MacGyver', 'Max_-_YouTube_Professional', 'Melina_CTC', 'Melissa_-_Female_Soothing_Narrator', 'Melville__Euro-accented_narrator.', 'Melvin_-_soothing_and_gentle', 'Mia', 'Mia_-_confident_and_annoyed', 'Mia_Chou', 'Michael', 'Michael_Anthony', 'Michelle_-_Old_and_Daring', 'Mike', 'Milan_Diekstra', 'Milean_-_bassy_with_plosives', 'Miller', 'Milo_-_Casual_Chill_Relatable_Young_Male', 'Mina', 'Miriam_-_Casual_and_Wry', 'Miss_Brittany_Andrews', 'Mkves_-_Calm_', 'Molly', 'Monika_Sogam', 'Mono', 'Morgan', 'Motivational_Coach_-_Leader', 'Mr._P_-_the_fun_guy', 'Mr_Novella_Main_Voice_-_Kobe_Black_British_Male_Young', 'Mun_W', 'My_Fortress', 'Nakiso', 'Nana-chan', 'Narender_Sharma', 'Narrador-34', 'Nata_Professional', 'Natasha_', 'Natasha_-_Sensual_Hypnosis', 'Nathaniel_C._-_Deep_Rich_Mature_British_voice', 'Naty_Heals_voice', 'Neal_-_Perfect_for_documentaries', 'Neil_-_cheerful_upbeat_youth', 'Nellie_-_soft', 'Newton', 'Niamh', 'Nichalia_Schwartz', 'Nick_Colter', 'Nicki', 'Nigel_-_Mysterious_Intriguing', 'Niladri_Mahapatra', 'Nina_-_nerdy', 'Nipunn_-_deep_captivating', 'Noah_-_scary_story_voice', 'Nora_', 'Nova_-_Wise_and_Tranquil_', 'Old_Joshua', 'Old_man_with_a_soft_voice', 'Olivia', 'Omeo', 'Oscar_-_Older_Narrative_Epic', 'Osiris_-_Deep_and_commanding_rumble_', 'Oswald_-_intelligent_professor', 'Pace_-_Deep_Menacing_and_Raspy', 'Page', 'Paladin', 'Parki_-_expressive_and_loud_elder', 'Paul_Henry_Smith_-_gentle_patient_clear', 'Paul_J._-_Calm_and_soothing_', 'Paula_Moon_-__Sleepy-time_true_crime_vocal', 'Paxti_-_Young_and_Earnest', 'Penelope_-_relaxed_and_breathy', 'Penfist_-_Military_Broadcaster', 'Penny_-_sweet_story_teller', 'Peter_-_Eastern_European_English', 'Peter_-_Hungarian_accent', 'Phil_-_Author_Non-fiction', 'Phillip', 'Phoebe', 'Pixy', 'Planty_-_raspy_voice', 'Prakash', 'Priya_-_Beautiful_and_melodic_Indian_accent', 'Priyam_-_Deep_Indian_Voice', 'Quasi-Jude-Lw', 'Rabih_Rizk', 'Rainbow', 'Raj', 'Raja_Babu', 'Ran', 'Rasper', 'Raven_Nightshade', 'Raven_Reed', 'Ray_-_Male_Soothing_Narrator', 'Raymond_Baxter', 'Raymond_Elliott', 'ReadingSam', 'Recvoice', 'RedGlassesVoiceovers', 'Remus_-_Fantasy_Professor', 'Rex_-_Throaty_and_World_Weary', 'Richard_-_enthusiastic_young_male', 'Richard_Yu', 'Ricky_The_K', 'Riley_-_loud_and_intense', 'Rinoa_-_Middle_Aged_Lady', 'Robert', 'Robert_-_American_standard_broadcaster', 'Robert_-_Business_Book_Narrator', 'Ronald_Wang', 'Rosalind_-_Classy_British_Actress', 'Rose', 'Rowan_-_gruff_and_raspy', 'Rufus', 'Rupert___Strong_British', 'Russel-_clear_realistic_pleasant', 'Russell', 'Russo_-_Dramatic_Australian_TV', 'Ryan', 'Ryan_-_Calm_Masculine_Teenager', 'Ryan_-_Dynamic_', 'Ryan_-_rough', 'Ryder_-_cool_and_balanced', 'SAVVAS', 'Sahand_RZ', 'Sally_Ford', 'Sam', 'Sam_-_English_Storyteller', 'Samantha', 'Sandy', 'Sanjana_', 'Sara_Jay', 'Sarah', 'Sarah_Lawson', 'Sash', 'Sassy_Aerisita', 'Satyam_1', 'Scar', 'Scott_-_Mature_and_Deep', 'Scott_-_Young_male_Canadian_voice', 'Scott_-_drill_instructor', 'Sean', 'Sean_-_deliberate_low_voice_authoritative_narration', 'Sean_Michael', 'Security', 'Serenity', 'Sexy_American_Female_voice', 'Seán', 'Shannon', 'Shannon_-_High_Quality_American_Male_Voice', 'Shanny_-_Soothing_Calm_American_Woman', 'Shelley_-_Clear_and_confident_British_female', 'Sheriff_Ben_-_Deep_Gruff_Authoritative', 'Shianne_-_Young_and_Confident', 'Shiv_-_Mature_Deep_Voice', 'Shoobu_-_Old_British_Man', 'Shot_List_voice_Girl', 'Sieu_Muoi', 'Sigrid_-_solemn_raspy_wise', 'Silas_-_stern_british_male', 'Silvia_-_upbeat_british_lady', 'Simeon', 'Simon_J_Kidson', 'Sina_-_Your_Narrator', 'SirEden_', 'Smart_Sara', 'Smarty_Pants_Amy', 'Smokey_McSmoker_-_Deep_and_Motivational', 'SocraGPTs', 'Sofy', 'Sophia', 'Sophia_Florence', 'Southern_Ann', 'Stan', 'Starry', 'Stella_-_Calm', 'Stephanie_P_-_Casual_feminine_great_for_storytelling', 'Stephen_-_Calm_British_Narrator_', 'Steve_Maughan', 'Steven_-_Calm_British_Deep_Soothing', 'Steven_-_Vibrant_Resonant_and_Inspiring', 'Stuart', 'Subirachs', 'Subu', 'Sully', 'Susan', 'Swara_-_Young__Calm_Voice', 'Tamara', 'Tanya-_Upbeat_and_Expressive', 'Tara', 'Tarini_-_Expressive__Cheerful_Narrator', 'Tarnish', 'Taro-_Young_Japanese_Accented_Guy', 'Tatsuya_Suzuki', 'Technical_Narrator_-_Precise_Knowledgeable_Engaging', 'Technical_Southerner', 'Temos_Sevandivasen_-_Resolute_Philosophical_Empathetic', 'Test_Aaron_2', 'Test_Plumb_2', 'Thaddeus_-_ancient_historian', 'Theodor_-_deep_american', 'Theodore-Old_Man__Deep_Husky_Voice', 'Theresa', 'Thomas', 'Tiffany_Kim_-_versatile_and_engaged_narrator', 'TikTok_Male_Voice', 'Todd_-_Universal_Crossover', 'Tom', 'Tom_-_trailer_narrator', 'Tommy_-_Reedy_Annoyed', 'Tony_-_Middle-aged_with_American_accent_', 'Trent_-_quirky', 'Tulipe', 'Twilight_Zone_Guy', 'Tyler_Kurk', 'Tyrone_-_Deep_Strong_Masculine_Narrator', 'Tyson', 'Upbeat_Teacher', 'Vee_-_Soft_Spoken_British_Male', 'Vicki_', 'Victoria_Queen_of_England', 'Vieux', 'Vivian', 'Vivie_2_Upbeat', 'W._Sillyman_Oxley', 'Wade_-_powerful', 'Walker', 'Wally_-_Warm_Deep_Masculine', 'Walter_-_Intelligent_and_Resolute', 'Wanda_-_calm_female', 'Wesley_-_nervous_cowardly_fellow', 'Will_', 'William', 'Winston_-_Distinguished_Erudite_and_Genteel', 'Winston__Authoritative_British_Man', 'Yee', 'Yoel', 'Young_brit', 'Yousef_-_Passionate_Sympathetic', 'Zara_-_understanding_friend', 'Zashikix', 'Zee_-_Childish', 'Zeus_-_arrogant', 'Zeus_Epic', 'Zoe_-_emphatic_and_pleasant', 'Zon-Kuthon', 'Zuri_-_New_Yorker', '_Luca_-_ calm_soothing_steady', '_Martha_-_Narration', '_Vicky_-_Posh_Voice_With_A_Lipse_', 'adriano_-_41', 'emily_', 'harry_deep_and_warm', 'madeline', 'neuris', 'sebastian_']

english2_names = ['19keys', 'ADAM', 'ADAM_v2', 'ALESSANDRO_DEEP', 'ATAKAN_ARISOY', 'A_Top_Narrator_VO_PRO', 'Aaditya_Kapur_-_Calm_Conversational_Voice', 'Aarav_-_Deep_and_wise_Indian', 'Aaron_-_Narration_Voice__A_Voice_thats_One_in_a_Million_NOT_like_a_Million_Others', 'Aaron_Davis_Emerson', 'Aaron_Sage_-_Friendly__Conversational_', 'Abigail', 'Adam_-_Calm_Smart', 'Adam_-_deep_voice_Australian', 'Adam_-_low_rough_and_full', 'Adam_-_old_and_knowing', 'Addie_-_Podcast_Princess', 'Adeline', 'Adi', 'Adriano_-_narrador_37', 'Aelar', 'Agatha', 'Agent_L', 'Ajay', 'Akua', 'Albert_-_Pleasant_deep_voice', 'Albert_-_Strong_German_Accent_', 'Albert_-_deep_slurred_meditations', 'Albert_Banoy', 'Alden_-_Resolute_Gravitas', 'Alex_-_Australian_Male_-_Casual_-_Melbourne_City', 'Alex_-_Business_Book_Narrator', 'Alex_-_Young_American_Male', 'Alex_-_expressive_narrator', 'Alex_Ozwyn', 'Alex_Wright', 'Alexander_-_Deep_Calm_and_Authoritative', 'Alexis_-_chic_and_cosmopolitan', 'Alfie', 'Ali', 'Alice_-_calm__composed', 'Alice_-_young_and_confident', 'Alita', 'Allison_-_millennial', 'Aly_-_Serious_and_Strict', 'Amaniri_-_British_Stalwart_Lass', 'Amelia_-_haughty', 'Amrut_Deshmukh_-_Booklet_Guy', 'Amy_-_Spunky_Cartoon_Girl_Voice', 'Amy_-_mean', 'Ana', 'Ana-Rita', 'Andrea_Wolff_-_clear_youthful_evenly_paced_', 'Andrew_-_tech_wizard', 'Andrews', 'Android_X.Y._Z._-_AI_Robot_of_the_Future', 'Angel', 'Ann_the_neighbor_', 'Anna_-_Cute_Calming_Narrator', 'Anne_Marie', 'Anthony10', 'Antoine', 'Antonio', 'Antonio_-_English_with_Subtle_Italian_Accent', 'Anup_Chugh_', 'Anushri_-_Natural_Young_Indian_Voice', 'Archie_-_English_teen_youth', 'Ardian', 'Ariah', 'Aristocrat', 'Arjun', 'Arnold', 'Arthur', 'Arthur_-_Energetic_American_Male_Narrator', 'Arthur_-_Geeky_Masculine_Deep', 'Arthur_-_Royal_Narrator', 'Arthur_the_anchorman', 'Arun', 'Ash', 'Asher_-_Confident_Aristocratic_Male', 'Ashley_American_Mom', 'Astrid', 'Astro_-_Audiobook_Excellence', 'Athena_-_corporate_supervisor', 'Aurelius_-_Calming_Deep_Serious_', 'Ava_Said_2', 'B._Hardscrabble_Oxley', 'Baron_Theatricus_-_Dramatic_Elocution', 'Beatrice_-_energetic_older_female_voice', 'Bedlam', 'Belinda', 'Ben_-_Masculine_Authorative', 'Ben_-_Scary_Stories', 'Benjamin', 'Benjamin_-_Deep_Warm_Calming', 'Benjamin_-_strong_and_confident', 'Bert_-_Mystical__Whimsical', 'Bill_Oxley_', 'Biquette_-_sad_and_resigned', 'Blake_-_bassy_and_gruff', 'Bob__-_Young_Deep-voice', 'Brian', 'Brian_-_Broadcast_News_Anchor', 'Brian_-_deep_narrator', 'Brian_Overturf', 'Brittany', 'Brittney', 'Brittney_-_Young_Peppy_Female_-_Social_Media_How_Tos_Explainers', 'Brody_-_Cool_Deep_Chilled', 'Bruce_-_Deep_Warm_Strong', 'Bruce_Actor', 'Brucifer', 'Brutus_-_Profound_Slow-paced_Inspiring', 'Bryan-Deep_Narration', 'Bryan_-_Narration', 'Bryan_-_Professional_Narrator', 'Bubba_Marshal', 'Burak_-_accented_storyteller', 'CAMILO_-_AMERICAN_VOICE_NARRATOR', 'CJ_Murph', 'CS_New_', 'Cal_-_Deep_and_Calming', 'Caleb', 'Cali_-_American_Female_voice_for_Promos', 'Cally_-_Young_and_Sweet', 'Calvin_', 'Cameron_-_deep_and_emotive', 'Can', 'Capt_Lynch_-_Sophisticated_Wise_Calm', 'Carlo_', 'Carlos_', 'CarterSutra', 'Carters_Edge', 'Casey_-_Clean_crisp_female_voice', 'Cecil_-_Profound_and_Precise', 'Cecilia', 'Charles_-_Deep_Hoarseness_Voice', 'Charlie_-_gentle_knowledgeable_old_voice___', 'Charlotte_-_sweet', 'Charmion_-_Soft_and_husky', 'Chloe_-_Girl_Next_Door', 'Chris_-_British_Friendly_Advertising_', 'Christian', 'Christina-_friendly_and_energetic', 'Christine_-_calm_teacher', 'Christopher', 'Christopher_-_Immersive', 'Christopher_-_scientific_mind', 'Chuck', 'Chuck_-_True_Crime', 'Claire', 'Clarice_-_Kind_and_Trustworthy', 'Clover_-_Calm_and_Collected', 'Cody_-_Authoritative__Deep_Motivational_Narration', 'Connor', 'Conny_-_Old_and_Stubborn', 'Consuelo', 'Conversational_Joe_-_A_chatty_casual_voice_British_RP_male', 'Courtney', 'Courtney_-_Soothing_and_Calm', 'Crime_Channel', 'Crystal_-_Pleasant_sultry_Voice_for_Audio_Experience', 'DJ_Marathon', 'DR_Dean_British', 'Dakota_H', 'Dalia_', 'Damon_-_Deep_and_Strong', 'Dan_-_Young_British_friendly_voice', 'Daniel_Lappisto', 'Daniel_R', 'Danielle_-_Canadian_Narrator', 'Dave', 'Dave_-_Dry_Quirky_Wit', 'David_-_American_Narrator', 'David_-_Epic_Movie_Trailer_', 'David_-_Gentle_Engaging_Soothing', 'David_-_Mature_Engaging_Male_Voice_American_accent', 'David_-_knowledgeable_old_soul_', 'Davy_-_Deep_Pirate_Voice', 'Dean_-_British_RP_Warm_and_Friendly', 'Dean_Jones', 'Delegate_-_Bright_and_Airy', 'Demeter_-_expressive_and_sincere_mother', 'Denis_-_Authoritative_and_Deep_Narrator', 'Denzel', 'Denzel_-_Casual_Narration_', 'Desdemona_-_balanced', 'Dez', 'Dispater_-_Refined_Strong__Authoritative_', 'Divija_-_A_female_voice_young_and_vibrant', 'Djali_Vesela', 'Don_-_Deep_Warm_Realistic', 'Don_Kim', 'Donald_-_American_70_years_old', 'Dorian_-_fast_paced_mediations', 'Dragonia_-_Dragon_Rider', 'Drake', 'Duncan_--_the_Melancholy_Intellectual_', 'Dying_story_teller', 'Ebony', 'Ebsa-_Realistic_Deep_male_voice', 'Ed_-_sweet_and_soft', 'Eddy', 'Edgar_-_nerdy', 'Edmund', 'Edris_-_deep_and_powerful', 'Edward-_muffled_and_distorted', 'Elaine_-_Sweet_and_Lively', 'Elaine_-_emotionally_versatile_narrator', 'Ele_-_Elegant_Youthful', 'Eli_-_American_voice_for_promos_and_explainers', 'Elijah_-_Narrative_Reader', 'Ella_', 'Ella_-_soft_and_sweet', 'Ellie_-_Tender_young_woman', 'Emily', 'Emily__-_pleasant_teen_voice', 'Emma_watson', 'Emms', 'Emre', 'Erecura_-_Walm_and_Nurturing', 'Erin', 'Erin_', 'Eris_-_strong', 'Eugene_-_nerd_and_geek', 'Evan_-_deep_narrator_voice', 'Evan_Byers', 'Evy_-_endearing_textured', 'Extraordinary_Joe_', 'Faheem_Ahmed_', 'Felicity_-_young_and_well-spoken', 'Finn_-_Serious_and_Sincere', 'Florence_-_Mature_Educated', 'Fowler_-_scary_and_authoratative', 'Franklin', 'Frederick_Surrey', 'Gandalf_', 'Garretts_Groove', 'Gemma_-_Refined_Witty_and_Warm', 'Gemma_-_Young_Australian_Female', 'Gene_-_informative_and_trustworthy', 'German_Petra_-_English_with_hard_accent', 'Giovanni', 'Godfrey_a_National_Treasure', 'Godot_-_Wise_and_Serene', 'Gordon_', 'Grace', 'GrandMaester_Game_of_Thrones', 'Grandma_Margaret_-_Storybook_Narrator', 'Grandpa_Slow_Reading', 'Grandpa_Spuds_Oxley', 'Gravitas_-_The_deep_narrator_voice', 'Gregory_-_British_Nature_Narrator', 'Gwen_-_Calm_and_Pleasant', 'Hades_-_grim_gravitas', 'Hamza', 'Hannah_-_assertive__refined', 'Hannah_-_soft-spoken', 'Harrison_-_Deep_and_Cinematic', 'Harrison_Gale_–_The_Velvet_Voice__deep_resonant_powerful_smooth_rich_storytelling_narrator', 'Haseeb_-_Canadian_Narration', 'Haven_Glass', 'Hector', 'Hector_-_Deep_Narrative', 'Hephaestus_-_steady_and_patient_teacher', 'Hiro', 'Hope_-_natural_conversations', 'Hope_-_upbeat_and_clear', 'Huss', 'Hyznberg_-_Crime_Time_Cool', 'Ian_Cartwell_-_Suspense_Mystery_and_Thriller', 'Igor_Radio', 'Illya_-_Soft_and_neutral', 'Isabel_-_Soft_Spoken_Teen_Youth', 'Isac', 'Isadore_', 'Ivy_-_Female_Childish_-_Young_Innocent__Bubbly', 'Ivys_Allure', 'Jace_Nox_-_Mellow_Gentle_and_Diverse', 'Jack_-_Calm_Monotone_Measured_Speech', 'Jack_the_Pirate', 'Jackson_-_Confident_Charismatic_and_Approachable', 'Jacob_-_Teen_and_Popular', 'Jacob_Dayi', 'Jacqui_Griffin', 'Jada_-_confident_and_direct', 'Jalia_-_soothing_female_voice', 'Jamal_', 'James', 'James_-_British_TV_presenter', 'James_-_classic_narrator', 'James_-_professional_and_authoritative', 'James_Lindsay_Pro', 'Jameson_-_Guided_Meditation__Narration', 'Jamie_-_young_child_voice', 'Jan', 'Janet', 'Jaquon', 'Jarvis_-_Polite_and_Upfront', 'Jason_-_Authoritative_Smooth_and_Approachable', 'Jay_-_Proper_Mancunian', 'Jeevan_-_Expressive_Indian_Voice', 'Jenny', 'Jerry_-_Presenter_Announcer_Event', 'Jerry_Beharry_-_Conversational', 'Jessica', 'Jessica_-_Cali_girl', 'Jessica_-_meditative', 'Jessica_-_smart_coach', 'Jessica_Anne_Bogart_-_A_VO_Professional_now_cloned', 'Jhon_-_casual_and_friendly', 'Jhonny_-_Agradable_reading', 'Joe', 'Joe_-_British_male_in_high_quality', 'Joey', 'Joey_-_Upbeat_Popular_News_Host', 'Joey_-_Youthful_and_Energetic', 'John2', 'John_-_American_War_Speech', 'John_-_Old_and_kind_', 'John_-_The_Heart_Of_America', 'John_Adams', 'John_Doe_-_Deep', 'John_Fernandes_-_Vibrant_British_Voice', 'John_Martin_-_Funny', 'Johnny_-_deep_and_gruff', 'Johnny_Lefors', 'Jona_-_man_of_the_Desert', 'Jonah_-_sassy_young_male', 'Jonathon', 'Jose_Feliciano_Voice_Clone', 'Joseph_-_Comforting_', 'Joseph_-_Cool_calm_and_great_for_narration', 'Joseph_-_motivational_speaker', 'Joy_Love', 'Judith_-_calm_and_confident', 'Julia_-_soft_and_shy', 'Julian_-_deep_rich_mature_British_voice', 'JuniorDT', 'Justin_-_hyped', 'Kai_Selekwa', 'Kamwe', 'Karan', 'Karan_-_EnglishStandup_Comedian_', 'Karl_C._Shroff_-_Professional_Calm_Voice', 'Karl_Stuke', 'Kass_-_Energetic_Casual_Engaging', 'Kat_Dollar', 'Kavya_-_Energetic_Kids_Voice_', 'Keel_-_confident_dramatic_narrator', 'Kelcey_-_Teen_and_Adventurous', 'Kellan_-_soft_and_gentle', 'Ken_-_African_man_with_heavy_accent', 'Ken_-_Influential_British_Male', 'Kenneth_-_calm_newcaster', 'Kevin_W._Krause_', 'Khemet_-_Deep_and_Powerful', 'Kostiantyn', 'Kyle_-_narrator', 'Lachita', 'Lana-_Robin_Rekia', 'Landon_Bailey', 'Laura_-_emphatic', 'Lauren_-Confident_Quick_Talking_No_Nonsense_Gal', 'Layo_Queen', 'Lee_-_Calm_and_Relaxed', 'Lena_-_emotive_and_expressive', 'Leo', 'Leo_-_Energetic_Indian_Voice', 'Leoni_Vergara_', 'Lily_Wolff_-_Expressive_Clear_Youthful_Calming', 'Lisa_-_pleasant', 'Lisa__-_Pleasant_calm_and_dynamic', 'Long_Storyteller', 'Lowy_-_soothing_gentle_and_warm', 'Lucan_Rook__-_Energetic_Male', 'Lucia_Reid_', 'Lucy_-_British_Storyteller', 'Lucy_-_sweet_and_sensual', 'Lucy_-_yound_anime_girl', 'Luis_-_Relaxed_and_Calm_Narration_-_Pro_Recording', 'Luke_old_and_deep', 'Luminessence_-_Light_Mirror', 'Luna_Spencer', 'Lydia_-_squeaky_', 'MW', 'Maccabaeus_-_Audiobook_Narration', 'Mael_-_deep_raspy_male', 'Mahmood', 'Manohar_-_Gruff_Seasoned_and_Wise', 'Marcus', 'Maria', 'Marie-Alice', 'Mark_-_Natural_Conversations', 'Mark_-_Robust_Dependable_and_Engaging', 'Mark_-_calm_and_wise_teacher', 'Marques_-_Young_and_Wary', 'Marshal_-_Grumpy_Sourpuss', 'Marshal_-_New_Jersey_Male', 'Martas', 'Mary_-_soft_and_warm', 'Matilda', 'Mats', 'Matt', 'Matt_Snowden', 'Matthew_-_calm_and_peaceful', 'Matthew_Wayne_-_Natural_calm_steady', 'Max_-_fast_friendly_and_direct', 'Maxwell_-_deep_and_dramatic', 'Maya', 'Meera', 'Melissa', 'Mellow_Matt', 'Merlin', 'Merlin_the_Wizard_Protector_of_King_Arthur', 'Mia_-_Clear_Smooth_Professional', 'Mia_-_Old_And_Confident_', 'Micah', 'Michael_', 'Michael_-_A_narrator_with_a_buttery-smooth_deep_voice', 'Michael_-_Confident', 'Michael_-_Excited_and_Ready_to_Speak', 'Michael_Filce', 'Michela', 'Middle_age_Southern_Male', 'Mike_Adams_-_All_things_space', 'Mike_G', 'Milo', 'Mine', 'Minerva_-_Fantasy_Professor', 'Mira_Gold_-_Dystopian', 'Miranda', 'Mirilene', 'Misti_-_English_Technology_Virtual_Training_Teacher', 'Mistress_Regina', 'Modavian_-_Dignified_Experienced_Authoritative', 'Moe', 'Mohammed', 'Mohanapriya', 'Monotone_Mike', 'Mora_of_Maragall_-_Resilient_Compassionate_Inspiring', 'Mouse', 'Mr_Clem', 'Mr_President_-_Strong_Fast_and_Impactful', 'Mrs_Novella_Main_Voice_-_Althea_Female_Young_European', 'Mwika_Kayange', 'Myriam_-_sweet_Teen_Girl', 'NEW_AMREEN', 'Nadya', 'Naina_-_Sophisticated_Indian_Girl', 'Nala_-_African_Female', 'Narrador_-_documentarios', 'Natalie_-_Posh_British', 'Natasha_-_Gentle_Meditation_', 'Natasha_-_Valley_girl', 'Nate_the_Great_-_American_Male', 'Neha', 'Neil', 'Neil_-_calm_and_deep', 'Neville', 'Nia_-_Black_Female', 'Niall_-_dramatic_male', 'Nichalia_Schwartz_-_Gentle_Kind_Sweet_GenAm', 'Nicholas_-_Raspy_Mature', 'Nickrad_', 'Nicola', 'Nicoletta', 'Nicolette_-_Strong_and_Stern', 'Nicolette_-_Young_Woman_Clear_Accented', 'Nigel_-_classic', 'Nigel_J.', 'Noah_-_The_stoic_narrator', 'Nolan_-_Emotive_and_Smooth', 'Northern_Irish_Peter', 'Northern_Terry', 'ONeal', 'Ocean_•_Monotonous_Voice', 'Oi', 'Okole', 'Old_Osirion_Woman_-_Timeless_Mystical_Nurturing', 'Older_British_gangster_-_Gravelly_and_Rough', 'Oliver_-_Documentary_Narration', 'Olivia-_sweet_and_soft', 'Omari_African_Voice_VERY_foreign_sounding.', 'Ophelia_Rose', 'Opsy', 'Oscar', 'Outstanding_for_Side_Character', 'Oxley_-_Evil_Character', 'Pablo_Marshal', 'Panda_Montana', 'Patino_-_Columbian_Spanish', 'Patrick_International', 'Patsy_Dahling', 'Paul_Martin', 'Pedro_Costa', 'Penelope', 'Persephone_-_lively', 'Peter', 'Peter_-_annoyingly_pitchy_and_enforcing', 'Peter_Owen_-_non-fiction_audiobooks_and_factual_VO', 'Philemon_-_serious_old_scientist', 'Pilar_-_Young_and_Cheerful', 'Piper', 'Pirate_Marshal', 'Pop', 'Pratheep_Tharan', 'Pro_Narrator_-_Convincing_story_teller', 'Prometheus', 'Queen_Rosamund_-_British_Older_Woman', 'Rachel', 'Rachel_M_-_Pro_British_Radio_Presenter_', 'Rachel__McGrath', 'Raju_-_Relatable_Indian_Voice', 'Rakhat_Eje_', 'Ralf_Eisend', 'Rama_-_wise_and_philosophical_sage', 'Randell-_Glone_Rekia', 'Randolph_-_Trustworthy_and_wise', 'Red_-_Dynamic_Expressive_and_Invigorating', 'Reginald_-_intense_villain', 'Researcher_-_Nerdy_and_Hesitant', 'Rhett_Sutton', 'Rhys_--_Sexy_British_Twink', 'Rich_Baritone_American_Radio_Announcer', 'Richard-2', 'Rike_Fischer', 'Road_Dawg__', 'Rob', 'Rob_-_confident_and_formal', 'Robert-__British_Narrator', 'Roberto_Riva', 'Ron_', 'Ron_-_Older_American_Story_Teller', 'Ruhaan_-_Clean_narration_voice', 'Rupert_-_British_60_years_old', 'Russell_-_Dramatic_British_TV', 'Ruth_-_grandmother_storyteller', 'Ryan_-_subtle_accent_and_deep_timbre', 'Ryan_Kurk', 'Ryan_Quin', 'Sagar_-_Voice_of_India', 'Sahara_-_Soothing_Meditation-Hypnosis-Romance', 'Sally', 'Sam_-_Chill_Southern_California_Male', 'Sam_-_Slight_Welsh_Accent', 'Sam_Bragg', 'Samantha_', 'Samantha_Narrations', 'Sanjay_-_profound_and_deep_', 'Sanna_Hartfield_Beta_1.0', 'Saphira_-_Teen_and_Nerdy', 'Sara', 'Sarcini_-_Snarky_Quick-witted_Unapologetic', 'Sasha_-_Soothing_and_Chill', 'Sayn_Awal', 'Scary_Story_', 'Scoobie_-_American_Male_enthusiastic_sharp_smart', 'Scot_Combs_Narration', 'Scott', 'Scott_Woodworth', 'Sean_John_-_Top_Quality', 'Sebastion-Young_uncertain._', 'Selena_-_Introspective_Intuitive', 'Seth_-_Vibrant_Engaging_and_Genuine', 'Sevan_Bomar_-_Black_Motivational_Speaker', 'Sexy_Female_Villain_Voice', 'Sgt_Hayes_-_Authority_Deep_Masculine', 'Shannon_B_-_Sad_Emo_Teenage_Girl', 'Shannon_B_-_Warm_Southern_Woman', 'Shayne_-_Narrator_RJ_Voice', 'Sheba', 'Shelby', 'Shelby_-_Erratic_and_Confident', 'Shells', 'Sheps_Rocky', 'Shrey_-_Deep__Engaging_', 'Simba', 'Sir_Linus_Warmheart', 'Sita_2', 'Soft_Daria_-_Meditation', 'Soft_Demure_Garden_Voice', 'Soft_young_male_voice', 'Sohaib_Jasra', 'Soothing_Narrator', 'Sophia_-_Female_UK_Accent_-_Audiobooks_E-learning_Courses_Adverts', 'Southern_Stewart', 'Sparrow_Lee_', 'Stanley', 'Stanley_', 'Starina_Jr_Pro', 'Stephen', 'Stephen_-_Narrator', 'Steve_-_Australian_Male_', 'Steve_-_British_-_Clean_Smooth_Professional', 'Steve_V', 'Steven', 'Steven_-_Business_Book_Narrator', 'Stuart_-_Energetic_and_enthusiastic', 'Summer', 'Sylvia_-_confident_sensible_wise', 'THE_PROTOTYPE_LIVE_aka_Ana_Daugherty', 'Tara_-_Conversational_Expressive_Voice', 'Tarun_C._Dhanraj_-_Rich_Warm_and_friendly', 'Tass', 'Tatiana', 'Taylor_Andrew_Commercial-Driven', 'Tere', 'Terry_Blackburn_', 'Tessa', 'Thalia_-_Mysteriously_Captivating', 'Thalias_Engine_-_Mysteriously_Captivating', 'The_Great_Conversationalist_', 'Theodore_-_Oldschool_Cool', 'Thomas_-_Measured_Clear_Informative_', 'Thomas_Candia', 'Thomas_Fischer_-_Authentic_German_Accent', 'Tiffany', 'Tim_Rooney', 'Tira_Shabbar_-_Spirited_Irreverent_Young-at-Heart', 'Tommy_-_Teen_Cool__Nonchalant_', 'Tony_-_King_of_New_York', 'Tony_-_middle_aged_male_Northern_English_native_accent', 'Tyler_', 'Tyrell', 'Tyrone', 'UK_Teen_-_Black_man_Marquess_Germain', 'Val_3.0', 'Valentino', 'Valentyna_-_Soft_and_calm', 'Veda', 'Very_Vlad_-_Soviet_Comrade', 'Victor_-_the_motivational_speaker', 'Victoria_-_classy_and_mature', 'Victorian_-_a_lady_of_quality', 'Victorino_-_Deep', 'Vidhi_-_Young__Bold', 'Vidura', 'Vikrant_-_Indian_', 'Vincent_Sparks_-_Deep_American_Voice', 'Vivian_-_knowledgeable_voice__', 'Von_-_Perfect_Storytelling_Clean_Realistic', 'W._Storytime_Oxley', 'Whimsy_-_Kids_Cartoon_Character', 'Whispering_Joe_-_a_storytelling_whisper_ASMR_British_RP_male', 'Wildebeest', 'Will_-_Young_Australian_Male', 'William_Shanks', 'Xanthippe_Abelló_-_Exuberant_Inquisitive_Unconventional', 'Yagiz', 'Yaisa', 'Yash_A_Malhotra_-_Warm__Friendly', 'Yomiee', 'Young_Jamal', 'Yuan_-_emotional_artist_poem_romantic_sensible', 'Zach_-_Storyteller_Narrator_Audiobooks_Podcasts', 'Zakirah_-_Chill_and_Calm', 'Zara_-_Soft_and_Serene_Indian_Voice', 'Zoe', 'Zoe_-_crisp_and_strong', '_Ethan_-_Calm_Intense_and_Compelling', '_Haseeb_-_Canadian_Presentation', '_Louis_Bloom', 'glenda-_soft_and_friendly', 'wise-woman']


# Función para cargar nombres en el dropdown según el idioma seleccionado
"""def load_names(selected_language):
    print(f"Idioma seleccionado: {selected_language}")
    if selected_language == "Arabic":
        return gr.update(choices=arabic_names, value=arabic_names[0] if arabic_names else None)
    elif selected_language == "Bulgarian":
        return gr.update(choices=bulgarian_names, value=bulgarian_names[0] if bulgarian_names else None)
    elif selected_language == "Chinese":
        return gr.update(choices=chinese_names, value=chinese_names[0] if chinese_names else None)
    else:
        return gr.update(choices=[], value=None)"""

def load_names(selected_language):
    print(f"Idioma seleccionado: {selected_language}")

    # Diccionario para mapear idiomas a listas de nombres
    nombres_por_idioma = {
        "Voices Legacy": show_legacy,
        "Arabic": arabic_names,
        "Bulgarian": bulgarian_names,
        "Chinese": chinese_names,
        "Croatian": croatian_names,
        "Czech": czech_names_names,  # Corregido el nombre de la variable
        "Danish": danish_names,
        "Dutch": dutch_names,
        "Finnish": finnish_names,
        "French": french_names,
        "German": german_names,
        "Greek": greek_names,
        "Hindi": hindi_names,
        "Hungarian": hungarian_names,
        "Indonesian": indonesian_names,
        "Italian": italian_names,
        "Japanese": japanese_names,
        "Korean": korean_names,
        "Norwegian": norwegian_names,
        "Polish": polish_names,
        "Portuguese": portuguese_names,
        "Romanian": romanian_names,
        "Russian": russian_names,
        "Slovak": slovak_names,
        "Spanish": spanish_names,
        "Swedish": swedish_names,
        "Tamil": tamil_names,
        "Turkish": turkish_names,
        "Ukrainian": ukrainian_names,
        "Vietnamese": vietnamese_names,
        "English-1": english1_names,
        "English-2": english2_names
    }

    nombres = nombres_por_idioma.get(selected_language, []) # Obtener la lista de nombres o una lista vacía si no se encuentra
    
    return gr.update(choices=nombres, value=nombres[0] if nombres else None)

def load_text(selected_name, selected_language):
    # Mapeo de idiomas a directorios
    directorios_por_idioma = {
        "Voices Legacy": "show_legacy",
        "Arabic": "ar",
        "Bulgarian": "bg",
        "Chinese": "zh",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English-1": "en1",  # Asegúrate de que estos nombres coincidan
        "English-2": "en2",  # con las claves de leng_and_ids
        "Finnish": "fi",
        "French": "fr",
        "German": "de",
        "Greek": "el",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Indonesian": "id",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Norwegian": "no",
        "Polish": "pl",
        "Portuguese": "pt",
        "Romanian": "ro",
        "Russian": "ru",
        "Slovak": "sk",
        "Spanish": "es",
        "Swedish": "sv",
        "Tamil": "ta",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Vietnamese": "vi"
    }
    dir_idioma = directorios_por_idioma.get(selected_language)
    if not dir_idioma:
        return "" # Manejar el caso en que el idioma no tenga directorio

    ruta_archivo = f"/tmp/Voice/{dir_idioma}/{selected_name}.txt"

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            texto = f.read()
        return texto
    except FileNotFoundError:
        return f"Archivo no encontrado: {ruta_archivo}"  # Mostrar un mensaje de error si el archivo no existe


def load_text_langs(selected_language):
    # Mapeo de idiomas a directorios
    directorios_por_idioma = {
        "Voices Legacy": "show_legacy",
        "Arabic": "ar",
        "Bulgarian": "bg",
        "Chinese": "zh",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English-1": "en1",  # Asegúrate de que estos nombres coincidan
        "English-2": "en2",  # con las claves de leng_and_ids
        "Finnish": "fi",
        "French": "fr",
        "German": "de",
        "Greek": "el",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Indonesian": "id",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Norwegian": "no",
        "Polish": "pl",
        "Portuguese": "pt",
        "Romanian": "ro",
        "Russian": "ru",
        "Slovak": "sk",
        "Spanish": "es",
        "Swedish": "sv",
        "Tamil": "ta",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Vietnamese": "vi"
    }
    dir_idioma = directorios_por_idioma.get(selected_language)

    return dir_idioma

# Función para cargar el texto y el audio de referencia
def update_reference_info(speaker_reference_audio, selected_language):
    # Actualizar el texto de referencia
    #print(speaker_reference_audio, load_text_langs(selected_language))
    text_info = load_text(speaker_reference_audio, selected_language)
    
    # Generar la ruta del archivo de audio
    audio_path = f"/tmp/Voice/{load_text_langs(selected_language)}/{speaker_reference_audio}.mp3"
    
    # Retornar ambos valores
    return text_info, audio_path

def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    vocab_path =  ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Params for TTS not found", "", "", ""         

    return "Params for TTS loaded", model_path, config_path, vocab_path,speaker_path, reference_path
     
def upload_audio(audio, current_path):
    if audio is None:
        return current_path

    upload_dir = "speaker_reference_audio"
    os.makedirs(upload_dir, exist_ok=True)

    if isinstance(audio, str):  # If it's a string (filepath)
        audio_path = audio  # Use the provided path directly
        if not os.path.exists(audio_path):  # Check if the file exists
            print(f"Error: File not found at {audio_path}")
            return current_path  # Or return an error message
        return audio_path     # Return the valid path


    elif hasattr(audio, "name"):  # If it's an UploadedFile object
        audio_path = os.path.join(upload_dir, audio.name)
        try:
            with open(audio_path, "wb") as f:
                f.write(audio.read())
            return audio_path
        except Exception as e:
            print(f"Error saving uploaded audio: {e}")
            return current_path
    else:
         print("The reference audio input format is not recognized")  
         return current_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        help="Name of the whisper model selected by default (Optional) Choices are: ['large-v3','large-v2', 'large', 'medium', 'small']   Default Value: 'large-v3'",
        default="large-v3",
    )
    parser.add_argument(
        "--audio_folder_path",
        type=str,
        help="Path to the folder with audio files (optional)",
        default="",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Enable sharing of the Gradio interface via public link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=7860,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: output/",
        default=str(Path.cwd() / "train_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 6",
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

    language_names = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "hu": "Hungarian",
    "ko": "Korean",
    "ja": "Japanese",
    }

    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink"), css = '''
    body {
        background-color: #333333;
        color: #E0E0E0;
    }
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .text-prompt {
        font-size: 1.2em;
        width: 200px; /* Ancho del checkbox */
        height: 20px; /* Alto del checkbox */
    }

    /* Estilo personalizado para el Toggle */
    .custom-toggle label {
        color: white; /* Color del texto */
    }
    .custom-toggle input[type="checkbox"] {
        appearance: none;
        -webkit-appearance: none;
        background-color: #181818; /* Color de fondo */
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        width: 20px; /* Ancho del checkbox */
        height: 20px; /* Alto del checkbox */
        border-radius: 5px; /* Bordes redondeados */
        display: inline-block;
        cursor: pointer;
        margin-right: 10px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .custom-toggle input[type="checkbox"]:checked {
        background-color: #FF0000; /* Cambio de azul a rojo */
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    .custom-toggle input[type="checkbox"]:hover {
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    /* Estilo para botones del menú */
    .menu-button {
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
        margin: 1px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.1s ease;
    }
    .menu-button:hover {
        background-color: #FF4500; /* Cambio de azul a naranja rojizo */
    }
    .menu-button:active {
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        transform: scale(0.95); /* Animación de clic */
    }
    /* Estilo para el botón activo */
    .menu-button.active {
        background-color: #FF0000; /* Cambio de azul a rojo */
    }
    /* Estilo para pestañas */
    .tab-nav {
        display: flex;
        justify-content: center;
        background-color: #242424;
        border-radius: 10px 10px 0 0; /* Bordes arriba redondeados */
        margin: 0;
    }
    .tab-button {
        background-color: black;
        color: white;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 10px 10px 0 0;  /* Bordes arriba redondeados */
        margin: 1px;
        transition: background-color 0.3s ease, transform 0.1s ease;
    }
    .tab-button:hover {
        background-color: #FF0000; /* Cambio de azul a rojo */
    }
    .tab-button:active {
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        transform: scale(0.95);
    }
    .tab-button.active {
        background-color: #FF0000; /* Cambio de azul a rojo */
    }
    /* Contenido de la pestaña */
    .tab-content {
        background-color: #242424;
        padding: 20px;
        border-radius: 0 0 50px 50px; /* Bordes abajo redondeados */
        border-top: none; /* Eliminar el borde superior */
    }
    .content {
        padding: 20px;
    }
    .video-container {
        text-align: center;
        padding: 10px;
    }
    .column {
        padding: 0px;
    }
    /* Estilo para los radio buttons */
    .custom-radio label {
        color: white;
    }
    .custom-radio input[type="radio"] {
        appearance: none;
        -webkit-appearance: none;
        background-color: #181818;
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-block;
        cursor: pointer;
        margin-right: 10px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .custom-radio input[type="radio"]:checked {
        background-color: #FF0000; /* Cambio de azul a rojo */
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    .custom-radio input[type="radio"]:hover {
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    /* Estilo para los sliders */
    .custom-slider input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        width: 100%;
        height: 10px;
        background: #181818;
        border-radius: 5px;
        outline: none;
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .custom-slider input[type="range"]:hover {
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    .custom-slider input[type="range"]:active {
        background: #8B0000; /* Cambio de azul a rojo oscuro */
    }
    .custom-slider::-webkit-slider-runnable-track {
        background-color: #FF0000; /* Cambio de azul a rojo */
        border-radius: 5px;
        height: 10px;
    }
    .custom-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        background-color: #FF0000; /* Cambio de azul a rojo */
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        width: 20px;
        height: 20px;
        border-radius: 50%;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    /* Estilo para los checkboxes */
    .custom-checkbox label {
        color: white;
    }
    .custom-checkbox input[type="checkbox"] {
        appearance: none;
        -webkit-appearance: none;
        background-color: #181818;
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        width: 20px;
        height: 20px;
        border-radius: 5px;
        display: inline-block;
        cursor: pointer;
        margin-right: 10px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .custom-checkbox input[type="checkbox"]:checked {
        background-color: #FF0000; /* Cambio de azul a rojo */
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    .custom-checkbox input[type="checkbox"]:hover {
        border-color: #FF0000; /* Cambio de azul a rojo */
    }
    /* Estilo para las imágenes */
    .custom-image {
        border: 2px solid #FF0000; /* Cambio de azul a rojo */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(255, 0, 0, 0.4); /* Cambio de azul a rojo */
    }

    /* Sidebar styles */
    .sidebar {
        width: 280px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .sidebar-header{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        gap: 10px;
    }

    .sidebar-header i{
        font-size: 1.2em;
    }

    .sidebar-header .badge{
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        color: white;
        padding: 4px 6px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .sidebar nav a {
        display: block;
        padding: 10px;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
        text-decoration: none;
        transition: background-color 0.3s;
        margin-bottom: 10px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .sidebar nav a:hover,
    .sidebar nav a.active {
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        color: white;
    }
    .prompt-section{
        margin-bottom: 20px;
    }
    .prompt-section label{
        display: block;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .prompt-section input{
        width: 100%;
        padding: 10px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        color: #FF4500; /* Cambio de azul a naranja rojizo */
        border: 1px solid #8B0000; /* Cambio de azul a rojo oscuro */
        border-radius: 6px;
        box-sizing: border-box;
    }

    .prompt-footer{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
    }

    .prompt-footer button{
        background: none;
        border: none;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
        font-size: 1.2em;
    }

    .prompt-error {
        display: flex;
        align-items: center;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 5px 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }

    .prompt-error i{
        color: #E57373; /* Rojo para el ícono de error */
        margin-right: 5px;
    }
    .generate-btn {
        background-color: #FF0000; /* Cambio de azul a rojo */
        color: white;
        padding: 12px 20px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        font-size: 1em;
    }

    .sidebar-footer{
        margin-top: 20px;
        font-size: 0.8em;
    }

    .itemized-bills{
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
        margin-top: 10px;
        cursor: pointer;
    }
    /* Main content styles */
    .main-content {
        flex: 1;
        padding: 20px;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    .top-header{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
    }

    .top-header div{
    border: 1px solid #8B0000; /* Cambio de azul a rojo oscuro */
    padding: 8px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8em;
    }
    .top-header .credits{
        display: flex;
        gap: 5px;
        align-items: center;
    }

    .top-header .notifications{
        position: relative;
    }
    .top-header .notifications .notification-dot{
        width: 8px;
        height: 8px;
        background-color: red;
        border-radius: 50%;
        position: absolute;
        top: 5px;
        right: 5px;
    }

    .main-header {
        font-size: 1.8em;
        margin-bottom: 20px;
    }

    .main-content-message{
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .progress-bar-container{
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }

    .queue-bar{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 10px;
    }

    .progress-line{
        flex: 1;
        height: 4px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .progress-dot{
        width: 12px;
        height: 12px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        border-radius: 50%;
        position: relative;
    }

    .progress-dot.active{
        background-color: #FF0000; /* Cambio de azul a rojo */
    }

    .queue-info{
        font-size: 0.9em;
        margin-bottom: 10px;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
    }

    .queue-info .community-link, .queue-info .upgrade-link{
        color: #FF0000; /* Cambio de azul a rojo */
        cursor: pointer;
        text-decoration: underline;
    }
    .upgrade-btn {
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        align-self: center;
        font-size: 1em;
    }
    .legal-notice{
        font-size: 0.7em;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
        display: flex;
        align-items: center;
        gap: 5px;
        margin-top: auto;
    }
    /* Video Queue styles */
    .video-queue {
        width: 300px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 20px;
        display: flex;
        flex-direction: column;
    }

    .video-queue-header{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .video-queue-header div{
        font-size: 1.1em;
        cursor: pointer;
    }

    .check-all{
        color: #FF4500; /* Cambio de azul a naranja rojizo */
    }
    #queue-list {
        /* display: flex;
        flex-direction: column; */
    }

    .video-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        border-radius: 6px;
        padding: 10px;
        cursor: pointer;
        gap: 10px;
        position: relative;
    }
    .video-item .generating-status{
        width: 30px;
        height: 30px;
        border: 4px solid #FF4500; /* Cambio de azul a naranja rojizo */
        border-radius: 50%;
        border-top-color: #FF0000; /* Cambio de azul a rojo */
        animation: loading 1s linear infinite;
    }

    .video-item .generating-text{
        font-size: 0.9em;
        color: #FF4500; /* Cambio de azul a naranja rojizo */
    }
    @keyframes loading {
        0%{
            transform: rotate(0deg);
        }
        100%{
            transform: rotate(360deg);
        }
    }
     .error-message{
        background-color: #8B0000; /* Cambio de azul a rojo oscuro */
        padding: 15px;
        border-radius: 6px;
        color: #E3F2FD;
        margin-bottom: 20px;
     }
'''
      ) as demo:
        gr.HTML(
        '''
        <div style="text-align: center; padding: 20px; background-color: #242424; border-radius: 10px;">
            <h1 style="font-size: 2.5em; color: #FF0000; margin: 0;">AI XTTS 5.3</h1>
            <p style="font-size: 1em; color: #E0E0E0; margin: 10px 0;">
                Created by: <a href="https://www.youtube.com/@IA.Sistema.de.Interes"
                target="_blank" style="color: #FF0000; text-decoration: none;">IA(Sistema de Interés)</a>
            </p>
        </div>
        '''
    )
        with gr.Tab("1 - Data processing"):
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=args.out_path,
            )
            # upload_file = gr.Audio(
            #     sources="upload",
            #     label="Select here the audio files that you want to use for XTTS trainining !",
            #     type="filepath",
            # )
            upload_file = gr.File(
                file_count="multiple",
                label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
            )
            
            audio_folder_path = gr.Textbox(
                label="Path to the folder with audio files (optional):",
                value=args.audio_folder_path,
            )

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                value=args.whisper_model,
                choices=[
                    "large-v3",
                    "large-v2",
                    "large",
                    "medium",
                    "small"
                ],
            )

            lang = gr.Dropdown(
                label="Dataset Language",
                value="en",
                choices=list(zip(language_names.values(), language_names.keys()))
            )
            progress_data = gr.Label(
                label="Progress:"
            )
            # demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset", elem_classes="menu-button")
        
            def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
            
                train_csv = ""
                eval_csv = ""
            
                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
            
                if audio_folder_path:
                    audio_files = list(list_audios(audio_folder_path))
                else:
                    audio_files = audio_path
            
                if not audio_files:
                    return "No audio files found! Please provide files via Gradio or specify a folder path.", "", ""
                else:
                    try:
                        # Loading Whisper
                        device = "cuda" if torch.cuda.is_available() else "cpu" 
                        
                        # Detect compute type 
                        if torch.cuda.is_available():
                            compute_type = "float16"
                        else:
                            compute_type = "float32"
                        
                        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path, gradio_progress=progress)
                    except:
                        traceback.print_exc()
                        error = traceback.format_exc()
                        return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""
            
                # clear_gpu_cache()
            
                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
                    print(message)
                    return message, "", ""
            
                print("Dataset Processed!")
                return "Dataset Processed!", train_meta, eval_meta


        with gr.Tab("2 - XTTS Encoder"):
            load_params_btn = gr.Button(value="Load Params from output folder", elem_classes="menu-button")
            version = gr.Dropdown(
                label="XTTS base version",
                value="v2.0.2",
                choices=[
                    "v2.0.3",
                    "v2.0.2",
                    "v2.0.1",
                    "v2.0.0",
                    "main"
                ],
            )
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            custom_model = gr.Textbox(
                label="(Optional) Custom model.pth file , leave blank if you want to use the base file.",
                value="",
            )
            num_epochs =  gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=2,
                maximum=512,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=2,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            clear_train_data = gr.Dropdown(
                label="Clear train data, you will delete selected folder, after optimizing",
                value="none",
                choices=[
                    "none",
                    "run",
                    "dataset",
                    "all"
                ])
            
            progress_train = gr.Label(
                label="Progress:"
            )

            # demo.load(read_logs, None, logs_tts_train, every=1)
            train_btn = gr.Button(value="Step 2 - Run the training", elem_classes="menu-button")
            optimize_model_btn = gr.Button(value="Step 2.5 - Optimize the model", elem_classes="menu-button")
            
            import os
            import shutil
            from pathlib import Path
            import traceback
            
            def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                clear_gpu_cache()
          
                # Check if `custom_model` is a URL and download it if true.
                if custom_model.startswith("http"):
                    print("Downloading custom model from URL...")
                    custom_model = download_file(custom_model, "custom_model.pth")
                    if not custom_model:
                        return "Failed to download the custom model.", "", "", "", ""
            
                run_dir = Path(output_path) / "run"
            
                # Remove train dir
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                
                # Check if the dataset language matches the language you specified 
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"
            
                # Check if lang.txt already exists and contains a different language
                current_language = None
                if lang_file_path.exists():
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()
                        if current_language != language:
                            print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                            language = current_language
                        
                if not train_csv or not eval_csv:
                    return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"The training was interrupted due to an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""
            
                ready_dir = Path(output_path) / "ready"
            
                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
            
                shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
            
                ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")
            
                # Move reference audio to output folder and rename it
                speaker_reference_path = Path(speaker_wav)
                speaker_reference_new_path = ready_dir / "reference.wav"
                shutil.copy(speaker_reference_path, speaker_reference_new_path)
            
                print("Model training done!")
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path

            def optimize_model(out_path, clear_train_data):
                # print(out_path)
                out_path = Path(out_path)  # Ensure that out_path is a Path object.
            
                ready_dir = out_path / "ready"
                run_dir = out_path / "run"
                dataset_dir = out_path / "dataset"
            
                # Clear specified training data directories.
                if clear_train_data in {"run", "all"} and run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {run_dir}: {e}")
            
                if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
                    try:
                        shutil.rmtree(dataset_dir)
                    except PermissionError as e:
                        print(f"An error occurred while deleting {dataset_dir}: {e}")
            
                # Get full path to model
                model_path = ready_dir / "unoptimize_model.pth"

                if not model_path.is_file():
                    return "Unoptimized model not found in ready folder", ""
            
                # Load the checkpoint and remove unnecessary parts.
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                del checkpoint["optimizer"]

                for key in list(checkpoint["model"].keys()):
                    if "dvae" in key:
                        del checkpoint["model"][key]

                # Make sure out_path is a Path object or convert it to Path
                os.remove(model_path)

                  # Save the optimized model.
                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
            
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)

                clear_gpu_cache()
        
                return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint

            def load_params(out_path):
                path_output = Path(out_path)
                
                dataset_path = path_output / "dataset"

                if not dataset_path.exists():
                    return "The output folder does not exist!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"

                # Write the target language to lang.txt in the output directory
                lang_file_path =  dataset_path / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()

                clear_gpu_cache()

                print(current_language)
                return "The data has been updated", eval_train, eval_csv, current_language




        with gr.Tab("3 - Inference"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button(value="Load params for TTS from output folder", elem_classes="menu-button")
                    xtts_checkpoint = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="XTTS config path:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="XTTS vocab path:",
                        value="",
                    )
                    xtts_speaker = gr.Textbox(
                        label="XTTS speaker path:",
                        value="",
                    )
                    progress_load = gr.Label(
                        label="Progress:"
                    )
                    load_btn = gr.Button(value="Step 3 - Load XTTS model", elem_classes="menu-button")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Speaker Reference Audio:",  # More descriptive label
                        value="",
                        interactive=True,  # Allow users to edit path manually
                    )
                    speaker_audio_upload = gr.Audio(
                        label="Upload Speaker Audio (wav, mp3, flac)",
                        type="filepath",  # Just keep type="filepath"
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        value="en",
                        choices=list(zip(language_names.values(), language_names.keys()))
                    )
                    tts_text = gr.Textbox(
                        label="Input Text.",
                        value="This model sounds really good and above all, it's reasonably fast.",
                    )
                    with gr.Accordion("Advanced settings", open=False) as acr:
                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.75,
                        )
                        length_penalty  = gr.Slider(
                            label="length_penalty",
                            minimum=-10.0,
                            maximum=10.0,
                            step=0.5,
                            value=1,
                        )
                        repetition_penalty = gr.Slider(
                            label="repetition penalty",
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=5,
                        )
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.85,
                        )
                        sentence_split = gr.Checkbox(
                            label="Enable text splitting",
                            value=True,
                        )
                        use_config = gr.Checkbox(
                            label="Use Inference settings from config, if disabled use the settings above",
                            value=False,
                        )
                    tts_btn = gr.Button(value="Step 4 - Inference", elem_classes="menu-button")
                    

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used.")


        with gr.Tab("2161 Voices"):
            with gr.Row():
                with gr.Column() as col1:
                    #load_params_tts_btn = gr.Button(value="Load params for TTS from output folder")
                    xtts_checkpoint0 = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="model/model.pth",
                    )
                    xtts_config0 = gr.Textbox(
                        label="XTTS config path:",
                        value="model/config.json",
                    )

                    xtts_vocab0 = gr.Textbox(
                        label="XTTS vocab path:",
                        value="model/vocab.json",
                    )
                    xtts_speaker0 = gr.Textbox(
                        label="XTTS speaker path:",
                        value="model/speakers_xtts.pth",
                    )
                    progress_load0 = gr.Label(
                        label="Progress:"
                    )
                    load_btn0 = gr.Button(value="Load model", elem_classes="menu-button")

                with gr.Column() as col2:

                    # Dropdown de selección de idioma
                    selected_language0 = gr.Dropdown(list(leng_and_ids.keys()), value="Select language", label="Language reference audio")
                    speaker_reference_audio0 = gr.Dropdown(interactive=True, allow_custom_value=True, label="Speaker reference audio:")
                    text_output0 = gr.Textbox(label="Audio reference information")

                    tts_language0 = gr.Dropdown(
                        label="Language",
                        value="en",
                        choices=list(zip(language_names.values(), language_names.keys()))
                    )
                    tts_text0 = gr.Textbox(
                        label="Input Text.",
                        value="This model sounds really good and above all, it's reasonably fast.",
                    )
                    with gr.Accordion("Advanced settings", open=False) as acr:
                        temperature0 = gr.Slider(
                            label="temperature",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.75,
                        )
                        length_penalty0  = gr.Slider(
                            label="length_penalty",
                            minimum=-10.0,
                            maximum=10.0,
                            step=0.5,
                            value=1,
                        )
                        repetition_penalty0 = gr.Slider(
                            label="repetition penalty",
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=5,
                        )
                        top_k0 = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                        )
                        top_p0 = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.85,
                        )
                        sentence_split0 = gr.Checkbox(
                            label="Enable text splitting",
                            value=True,
                        )
                        use_config0 = gr.Checkbox(
                            label="Use Inference settings from config, if disabled use the settings above",
                            value=False,
                        )
                    tts_btn0 = gr.Button(value="Generate", elem_classes="menu-button")
                  

                    selected_language0.change(load_names, inputs=selected_language0, outputs=speaker_reference_audio0)
                    #speaker_reference_audio0.change(load_text, inputs=[speaker_reference_audio0, selected_language0], outputs=text_output0)
                    

                with gr.Column() as col3:
                    progress_gen0 = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio0 = gr.Audio(label="Generated Audio.")
                    reference_audio0 = gr.Audio(label="Reference audio used.")
                    speaker_reference_audio0.change(update_reference_info, inputs=[speaker_reference_audio0, selected_language0], outputs=[text_output0, reference_audio0])





            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    audio_folder_path,
                    lang,
                    whisper_model,
                    out_path,
                    train_csv,
                    eval_csv
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )


            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[
                    progress_train,
                    train_csv,
                    eval_csv,
                    lang
                ]
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    custom_model,
                    version,
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[
                    out_path,
                    clear_train_data
                ],
                outputs=[progress_train,xtts_checkpoint0],
            )
            

            load_btn0.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint0,
                    xtts_config0,
                    xtts_vocab0,
                    xtts_speaker0
                ],
                outputs=[progress_load0],
            )

            tts_btn0.click(
                fn=run_tts0,
                inputs=[
                    selected_language0,
                    tts_language0,
                    tts_text0,
                    speaker_reference_audio0,
                    temperature0,
                    length_penalty0,
                    repetition_penalty0,
                    top_k0,
                    top_p0,
                    sentence_split0,
                    use_config0
                ],
                outputs=[progress_gen0, tts_output_audio0,reference_audio0],
            )

            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker
                ],
                outputs=[progress_load],
            )



            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                    temperature,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config
                ],
                outputs=[progress_gen, tts_output_audio,reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[
                    out_path,
                    version
                    ],
                outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
            )

            speaker_audio_upload.upload(
                upload_audio,
                inputs=[speaker_audio_upload, speaker_reference_audio],
                outputs=speaker_reference_audio,
            )


             

    demo.launch(
        share=args.share,
        debug=False,
        server_port=7860,
        # inweb=True,
        server_name="0.0.0.0"
    )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import speech_recognition as sr
import nltk
from nltk.corpus import cmudict
import tempfile
import base64

# Download CMU Pronouncing Dictionary for English
nltk.download("cmudict", quiet=True)
pron_dict = cmudict.dict()

# Initialize recognizer class
recognizer = sr.Recognizer()

# Language mapping for speech recognition
LANGUAGE_CODES = {
    "English": "en-US",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Russian": "ru-RU",
    "Japanese": "ja-JP",
    "Hindi": "hi-IN"
}

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- MODIFIED FUNCTIONS FOR CLOUD ---

# Remove PyAudio recording function and replace with file upload
def record_audio_cloud():
    """Cloud-compatible audio input via file upload"""
    st.warning("⚠️ Live microphone recording not available in cloud deployment")
    st.info("Please upload an audio file (WAV format) or use text input")
    
    uploaded_file = st.file_uploader("Upload Audio File (WAV format)", type=['wav'])
    
    if uploaded_file is not None:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(uploaded_file.getvalue())
            temp_filename = temp_audio.name
        
        # Display audio player
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format='audio/wav')
        
        return temp_filename
    return None

# Keep the speech_to_text function but modify it for file upload
def speech_to_text(audio_file, language="en-US"):
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
            # Use Google Speech Recognition API
            text = recognizer.recognize_google(audio_data, language=language)
            return text
    except sr.UnknownValueError:
        return "Speech not recognized. Please try again."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"Error processing audio: {e}"
    finally:
        # Clean up the temporary file
        try:
            os.remove(audio_file)
        except:
            pass

# Keep the rest of your functions as they are (text_to_phonemes, compare_phonemes, etc.)
# BUT remove Levenshtein dependency for cloud compatibility
def compare_phonemes_cloud(expected_phonemes, spoken_phonemes):
    """Simplified comparison without Levenshtein for cloud"""
    total_accuracy = 0
    total_words = len(expected_phonemes)
    word_feedback = {}

    for word, expected in expected_phonemes.items():
        spoken = spoken_phonemes.get(word, ["<UNK>"])
        
        # Simplified accuracy calculation
        if spoken == ["<UNK>"]:
            accuracy = 0
        elif spoken == expected:
            accuracy = 100
        else:
            # Count matching phonemes
            matches = sum(1 for e, s in zip(expected, spoken) if e == s)
            accuracy = (matches / max(len(expected), len(spoken))) * 100
        
        total_accuracy += accuracy

        word_feedback[word] = {
            "Expected": " ".join(expected),
            "Spoken": " ".join(spoken),
            "Accuracy": round(accuracy, 2),
            "Correction": f"Try saying '{word}' as /{' '.join(expected)}/" if accuracy < 80 else "✅ Good pronunciation!"
        }

    overall_accuracy = round((total_accuracy / total_words), 2) if total_words > 0 else 0
    return overall_accuracy, word_feedback

# Keep the rest of your functions (calculate_fluency, plot_side_by_side_charts, etc.)
# [PASTE ALL YOUR OTHER FUNCTIONS HERE - plot_side_by_side_charts, plot_trend_graph, etc.]

# Streamlit UI - MODIFIED FOR CLOUD
st.title("🌎 Pronunciation Analyzer & Coach (Cloud Version)")

# Language selection
selected_language = st.selectbox(
    "Select Language:",
    list(LANGUAGE_CODES.keys())
)

# Update session state
if selected_language != st.session_state.get('current_language', 'English'):
    st.session_state.current_language = selected_language

# Create tabs
tab1, tab2 = st.tabs(["Pronunciation Analyzer", "Pronunciation Assistant Chat"])

with tab1:
    # Input reference text
    EXAMPLE_TEXTS = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
        "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Japanese": "素早い茶色のキツネは怠惰な犬を飛び越える。",
        "Hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"
    }
    
    reference_text = st.text_area(
        "Enter the reference text:", 
        EXAMPLE_TEXTS.get(selected_language, "The quick brown fox jumps over the lazy dog.")
    )
    
    # CLOUD-FRIENDLY INPUT OPTIONS
    st.subheader("Choose Input Method:")
    
    input_method = st.radio(
        "Select how to provide your speech:",
        ["Upload Audio File", "Enter Text Manually", "Demo Mode"]
    )
    
    spoken_text = ""
    audio_file = None
    
    if input_method == "Upload Audio File":
        audio_file = record_audio_cloud()
        if audio_file and st.button(" Analyze Audio"):
            with st.spinner("Processing audio..."):
                language_code = LANGUAGE_CODES.get(selected_language, "en-US")
                spoken_text = speech_to_text(audio_file, language=language_code)
                
    elif input_method == "Enter Text Manually":
        spoken_text = st.text_area("Enter what you said:", height=100)
        if st.button(" Analyze Text"):
            if spoken_text:
                st.success("Text submitted for analysis!")
            else:
                st.error("Please enter some text to analyze.")
                
    elif input_method == "Demo Mode":
        spoken_text = reference_text  # Use reference text as demo
        if st.button(" Run Demo"):
            st.info("Running in demo mode - using reference text as spoken text")
    
    # ANALYSIS SECTION
    if spoken_text and not ("Speech not recognized" in spoken_text or "Error" in spoken_text):
        # Extract phonemes
        expected_phonemes = text_to_phonemes(reference_text, selected_language)
        spoken_phonemes = text_to_phonemes(spoken_text, selected_language)

        # Pronunciation accuracy and correction suggestions
        overall_accuracy, word_feedback = compare_phonemes_cloud(expected_phonemes, spoken_phonemes)

        # Fluency scoring (use default duration for demo)
        fluency = calculate_fluency(reference_text, duration=5)

        # Store scores for trend graph
        if 'fluency_scores' not in st.session_state:
            st.session_state.fluency_scores = []
        if 'pronunciation_accuracies' not in st.session_state:
            st.session_state.pronunciation_accuracies = []
            
        st.session_state.fluency_scores.append(fluency)
        st.session_state.pronunciation_accuracies.append(overall_accuracy)
        
        # Store last analysis results for chatbot
        st.session_state.last_word_feedback = word_feedback
        st.session_state.last_overall_accuracy = overall_accuracy
        st.session_state.last_fluency = fluency
        st.session_state.last_spoken_text = spoken_text
        st.session_state.last_reference_text = reference_text

        # Display results
        st.subheader(" Pronunciation Analysis")
        st.write(f" **You Said:** {spoken_text}")
        st.write(f" **Overall Pronunciation Accuracy:** {overall_accuracy}%")
        st.write(f"⚡ **Fluency Score:** {round(fluency, 2)}%")

        # Generate and display chatbot response
        chatbot_response = generate_chatbot_response(
            overall_accuracy, fluency, word_feedback, spoken_text, reference_text, selected_language
        )
        st.session_state.chat_history.append(("Assistant", chatbot_response))
        
        # Show Graphs side by side
        st.subheader(" Visualization")
        fig = plot_side_by_side_charts(fluency, overall_accuracy, word_feedback)
        st.pyplot(fig)
        
        # Show trend graph if multiple attempts exist
        if len(st.session_state.fluency_scores) > 1:
            st.subheader(" Progress Trends")
            trend_fig = plot_trend_graph(st.session_state.fluency_scores, st.session_state.pronunciation_accuracies)
            st.pyplot(trend_fig)

        # Word-by-word analysis
        st.subheader("🔍 Word-by-Word Feedback")
        feedback_data = [
            [word, f"{feedback['Accuracy']}%", feedback["Expected"], feedback["Spoken"], feedback["Correction"]]
            for word, feedback in word_feedback.items()
        ]
        df = pd.DataFrame(feedback_data, columns=["Word", "Accuracy", "Expected Phonemes", "Spoken Phonemes", "Correction Suggestion"])
        st.table(df)

with tab2:
    st.subheader(f"💬 Chat with Your {selected_language} Pronunciation Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for speaker, message in st.session_state.chat_history:
            if speaker == "User":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")
    
    # Chat input
    st.text_input(f"Ask a question about your {selected_language} pronunciation or get tips:", key="chat_input", on_change=send_message)
    
    # Show some example questions to help the user
    with st.expander("Example questions you can ask"):
        st.markdown("""
        - "What were my worst pronounced words?"
        - "How can I improve my fluency?"
        - "Give me tips to improve my pronunciation"
        - "What is a phoneme?"
        - "What was my accuracy for specific words?"
        - "How do I improve my accent in this language?"
        """)

# Add a language help section
with st.expander("Language-Specific Tips"):
    if selected_language == "English":
        st.markdown("""
        ### English Pronunciation Tips
        - Focus on the 'th' sound which is uncommon in many languages
        - Pay attention to word stress patterns
        - Practice vowel sounds, especially those that don't exist in your native language
        - Work on the difference between similar sounds like 'r' and 'l'
        """)
    elif selected_language == "Spanish":
        st.markdown("""
        ### Spanish Pronunciation Tips
        - Practice rolling your 'r' sounds
        - Pay attention to the difference between 'b' and 'v'
        - Learn the correct pronunciation of 'h' (silent) and 'j' (harsh sound)
        - Work on the rhythm and stress patterns
        """)
    elif selected_language == "French":
        st.markdown("""
        ### French Pronunciation Tips
        - Practice nasal vowel sounds
        - Work on the French 'r' sound
        - Pay attention to liaison (connecting words)
        - Focus on proper stress which generally falls on the last syllable
        """)
    # Add other languages as needed

# Add information about how the app works
with st.expander("How this app works"):
    st.markdown("""
    ### Pronunciation Analysis System
    
    This app uses speech recognition and phonetic analysis to help you improve your pronunciation in multiple languages:
    
    1. **Speech Recognition**: The app uses Google's Speech Recognition API to convert your speech to text
    
    2. **Phonetic Analysis**: For English, we use the CMU Pronouncing Dictionary to break words into their component sounds. For other languages, we use approximation techniques
    
    3. **Accuracy Measurement**: We compare the expected phonetic representation with what you actually said
    
    4. **Fluency Assessment**: We measure your speaking pace and rhythm to evaluate fluency
    
    5. **Visual Feedback**: Charts and tables show you exactly where to improve
    
    6. **Smart Assistant**: Our AI assistant provides personalized feedback and tips in your chosen language
    
    For best results, speak clearly in a quiet environment, and practice regularly with a variety of texts.
    """)


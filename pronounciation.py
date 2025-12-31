import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pyaudio
import wave
import os
import speech_recognition as sr
import nltk
from nltk.corpus import cmudict
import Levenshtein
import time
from io import BytesIO
import tempfile

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

# Function to capture live speech
def record_audio(duration=5, sample_rate=16000, channels=1):
    st.write("🎙️ Recording... Speak now!")
    
    audio_format = pyaudio.paInt16
    chunk = 1024
    audio = pyaudio.PyAudio()
    
    # Start recording
    stream = audio.open(format=audio_format,
                       channels=channels,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=chunk)
    
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save recording temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_filename = temp_audio.name
        
        # Save as WAV file
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    return temp_filename

# Convert speech to text using SpeechRecognition library
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

# Convert text to phonemes with language-specific handling
def text_to_phonemes(text, language):
    words = text.lower().split()
    
    if language == "English":
        # Use CMU dictionary for English
        return {word: pron_dict.get(word, [["<UNK>"]])[0] for word in words}
    else:
        # For other languages, use a simpler phonetic approximation
        # We'll use characters as a basic phonetic representation
        return {word: list(word) for word in words}

# Compare phonemes and suggest corrections
def compare_phonemes(expected_phonemes, spoken_phonemes):
    total_accuracy = 0
    total_words = len(expected_phonemes)
    word_feedback = {}

    for word, expected in expected_phonemes.items():
        spoken = spoken_phonemes.get(word, ["<UNK>"])
        edit_distance = Levenshtein.distance(" ".join(expected), " ".join(spoken))
        max_length = max(len(expected), len(spoken))
        accuracy = 1 - (edit_distance / max_length) if max_length > 0 else 0
        total_accuracy += accuracy

        word_feedback[word] = {
            "Expected": " ".join(expected),
            "Spoken": " ".join(spoken),
            "Accuracy": round(accuracy * 100, 2),
            "Correction": f"Try saying '{word}' as /{' '.join(expected)}/" if accuracy < 0.8 else "✅ Good pronunciation!"
        }

    overall_accuracy = round((total_accuracy / total_words) * 100, 2) if total_words > 0 else 0
    return overall_accuracy, word_feedback

# Calculate fluency score
def calculate_fluency(reference_text, duration):
    words_per_minute = (len(reference_text.split()) / duration) * 60
    return min(100, words_per_minute / 150 * 100)  # Normalize to percentage

# Function to plot charts side by side
def plot_side_by_side_charts(fluency_score, pronunciation_accuracy, word_feedback):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart on the left
    labels = ["Fluency Score", "Pronunciation Accuracy"]
    sizes = [max(0, fluency_score), max(0, pronunciation_accuracy)]  # Ensure non-negative
    
    if sum(sizes) == 0:
        ax1.text(0.5, 0.5, "⚠️ Not enough data for pie chart", ha='center', va='center')
    else:
        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"])
        ax1.set_title("Fluency vs. Pronunciation Accuracy")
    
    # Bar chart on the right
    words = list(word_feedback.keys())
    accuracies = [word_feedback[word]["Accuracy"] for word in words]
    
    bars = ax2.bar(words, accuracies, color=["green" if acc > 80 else "red" for acc in accuracies])
    ax2.set_xlabel("Words")
    ax2.set_ylabel("Pronunciation Accuracy (%)")
    ax2.set_title("Word-by-Word Pronunciation Accuracy")
    ax2.set_ylim(0, 100)  # Set y-axis from 0-100%
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Function to plot trend graph - FIXED FUNCTION
def plot_trend_graph(fluency_scores, pronunciation_accuracies):
    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = list(range(1, len(fluency_scores) + 1))

    ax.plot(x_axis, fluency_scores, marker="o", linestyle="-", color="blue", label="Fluency Score")
    ax.plot(x_axis, pronunciation_accuracies, marker="s", linestyle="-", color="green", label="Pronunciation Accuracy")

    ax.set_xlabel("Attempts")
    ax.set_ylabel("Scores (%)")
    ax.set_title("Pronunciation & Fluency Trend Over Time")
    ax.legend()
    
    return fig  # Return the figure object directly

# Language-specific feedback phrases
FEEDBACK_PHRASES = {
    "English": {
        "excellent": "Great job! Your pronunciation is excellent.",
        "good": "Good effort! Your pronunciation is pretty good, but there's room for improvement.",
        "needs_work": "Let's work on your pronunciation. I've identified some areas where you can improve.",
        "fluency_excellent": "Your speaking pace is excellent!",
        "fluency_good": "Your speaking pace is good, but try to be more fluid.",
        "fluency_needs_work": "Try to speak a bit more smoothly and maintain a steady pace.",
        "focus_words": "Focus on these words:",
        "missing_words": "You missed these words:",
        "missing_words_many": "You missed {count} words from the reference text.",
        "added_words": "You added these extra words:",
        "added_words_many": "You added {count} extra words that weren't in the reference text.",
        "closing": "Keep practicing! Would you like to try again or get tips for any specific word?"
    },
    "Spanish": {
        "excellent": "¡Excelente trabajo! Tu pronunciación es excelente.",
        "good": "¡Buen esfuerzo! Tu pronunciación es bastante buena, pero hay margen de mejora.",
        "needs_work": "Trabajemos en tu pronunciación. He identificado algunas áreas en las que puedes mejorar.",
        "fluency_excellent": "¡Tu ritmo de habla es excelente!",
        "fluency_good": "Tu ritmo de habla es bueno, pero intenta ser más fluido.",
        "fluency_needs_work": "Intenta hablar un poco más suavemente y mantener un ritmo constante.",
        "focus_words": "Concéntrate en estas palabras:",
        "missing_words": "Omitiste estas palabras:",
        "missing_words_many": "Omitiste {count} palabras del texto de referencia.",
        "added_words": "Añadiste estas palabras extra:",
        "added_words_many": "Añadiste {count} palabras extra que no estaban en el texto de referencia.",
        "closing": "¡Sigue practicando! ¿Te gustaría intentarlo de nuevo o recibir consejos para alguna palabra específica?"
    },
    # Add other languages as needed
}

# Generate chatbot response based on pronunciation analysis with language support
def generate_chatbot_response(overall_accuracy, fluency, word_feedback, spoken_text, reference_text, language):
    # Get language-specific phrases (default to English if not found)
    phrases = FEEDBACK_PHRASES.get(language, FEEDBACK_PHRASES["English"])
    
    # Start with a general assessment
    if overall_accuracy > 90:
        response = phrases["excellent"] + " "
    elif overall_accuracy > 70:
        response = phrases["good"] + " "
    else:
        response = phrases["needs_work"] + " "
    
    # Add fluency comment
    if fluency > 90:
        response += phrases["fluency_excellent"] + " "
    elif fluency > 70:
        response += phrases["fluency_good"] + " "
    else:
        response += phrases["fluency_needs_work"] + " "
    
    # Add specific word feedback for the worst-pronounced words
    problem_words = [word for word, data in word_feedback.items() if data["Accuracy"] < 70]
    if problem_words:
        response += f"\n\n{phrases['focus_words']} "
        for word in problem_words[:3]:  # Limit to 3 worst words
            response += f"\n- '{word}': {word_feedback[word]['Correction']}"
    
    # Check for missing or added words
    reference_words = set(reference_text.lower().split())
    spoken_words = set(spoken_text.lower().split())
    
    missing_words = reference_words - spoken_words
    if missing_words and len(missing_words) <= 3:
        response += f"\n\n{phrases['missing_words']} {', '.join(missing_words)}."
    elif missing_words:
        response += f"\n\n{phrases['missing_words_many'].format(count=len(missing_words))}"
    
    added_words = spoken_words - reference_words
    if added_words and len(added_words) <= 3:
        response += f"\n\n{phrases['added_words']} {', '.join(added_words)}."
    elif added_words:
        response += f"\n\n{phrases['added_words_many'].format(count=len(added_words))}"
    
    # Add an encouraging closing statement
    response += f"\n\n{phrases['closing']}"
    
    return response

# Store trends across multiple attempts
if 'fluency_scores' not in st.session_state:
    st.session_state.fluency_scores = []
if 'pronunciation_accuracies' not in st.session_state:
    st.session_state.pronunciation_accuracies = []
if 'last_word_feedback' not in st.session_state:
    st.session_state.last_word_feedback = {}
if 'last_overall_accuracy' not in st.session_state:
    st.session_state.last_overall_accuracy = 0
if 'last_fluency' not in st.session_state:
    st.session_state.last_fluency = 0
if 'last_spoken_text' not in st.session_state:
    st.session_state.last_spoken_text = ""
if 'last_reference_text' not in st.session_state:
    st.session_state.last_reference_text = ""
if 'current_language' not in st.session_state:
    st.session_state.current_language = "English"
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

# Function to handle user chat input
def handle_chat_input(user_input, language):
    if not user_input.strip():
        return "Please type something or ask a question."
    
    # Get language-specific phrases
    phrases = FEEDBACK_PHRASES.get(language, FEEDBACK_PHRASES["English"])
    
    # If we have pronunciation data, use it to provide personalized responses
    if st.session_state.last_word_feedback:
        # Check for questions about specific words
        user_words = user_input.lower().split()
        for word in user_words:
            if word in st.session_state.last_word_feedback:
                feedback = st.session_state.last_word_feedback[word]
                return f"For the word '{word}', try saying it as /{feedback['Expected']}/. Your pronunciation accuracy was {feedback['Accuracy']}%."
        
        # Check for common questions
        if "worst" in user_input.lower() or "difficult" in user_input.lower():
            worst_words = sorted(st.session_state.last_word_feedback.items(), key=lambda x: x[1]["Accuracy"])[:3]
            if worst_words:
                response = "Your most difficult words were:\n"
                for word, data in worst_words:
                    response += f"- '{word}' ({data['Accuracy']}% accuracy): Try saying it as /{data['Expected']}/\n"
                return response
            else:
                return "I don't have enough data yet to identify your most difficult words."
        
        if "best" in user_input.lower() or "good" in user_input.lower():
            best_words = sorted(st.session_state.last_word_feedback.items(), key=lambda x: x[1]["Accuracy"], reverse=True)[:3]
            if best_words:
                response = "Your best pronounced words were:\n"
                for word, data in best_words:
                    response += f"- '{word}' ({data['Accuracy']}% accuracy)\n"
                return response
            else:
                return "I don't have enough data yet to identify your best pronounced words."
        
        if "tips" in user_input.lower() or "advice" in user_input.lower() or "improve" in user_input.lower():
            if st.session_state.last_overall_accuracy < 70:
                return "To improve your pronunciation:\n1. Listen carefully to native speakers\n2. Practice individual sounds that you find difficult\n3. Record yourself and compare to reference audio\n4. Slow down and focus on accuracy before speed\n5. Try tongue twisters to improve articulation"
            else:
                return "Your pronunciation is already quite good! To reach the next level:\n1. Focus on rhythm and intonation\n2. Try speaking at a natural pace\n3. Record longer passages to practice connected speech\n4. Pay attention to word stress and syllable emphasis"
    
    # General responses if no pronunciation data is available or for other questions
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return "Hello! I'm your pronunciation assistant. Try recording yourself reading the text, and I'll analyze your pronunciation and provide feedback."
    
    if "how" in user_input.lower() and "work" in user_input.lower():
        return "I analyze your pronunciation by comparing the phonemes (speech sounds) of what you say to the expected phonemes. I also measure your fluency based on your speaking pace. Click 'Start Live Speech' to try it out!"
    
    # Default response
    return "I'm here to help with your pronunciation. Try recording yourself reading the text, ask about specific words, or ask for tips to improve your pronunciation."

# Function to handle sending a message
def send_message():
    user_message = st.session_state.chat_input
    if user_message:
        st.session_state.chat_history.append(("User", user_message))
        assistant_response = handle_chat_input(user_message, st.session_state.current_language)
        st.session_state.chat_history.append(("Assistant", assistant_response))
        st.session_state.chat_input = ""  # Clear the input

# Example texts for each language
EXAMPLE_TEXTS = {
    "English": "The quick brown fox jumps over the lazy dog.",
    "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
    "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
    "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Japanese": "素早い茶色のキツネは怠惰な犬を飛び越える。",
    "Hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"
}

# Streamlit UI
st.title("🌎 Pronunciation Analyzer & Coach")

# Language selection
selected_language = st.selectbox(
    "Select Language:",
    list(LANGUAGE_CODES.keys())
)

# Update current language if changed
if selected_language != st.session_state.current_language:
    st.session_state.current_language = selected_language

# Create tabs for the main app sections
tab1, tab2 = st.tabs(["Pronunciation Analyzer", "Pronunciation Assistant Chat"])

with tab1:
    # Input reference text with language-specific example
    reference_text = st.text_area(
        "Enter the reference text:", 
        EXAMPLE_TEXTS.get(selected_language, "The quick brown fox jumps over the lazy dog.")
    )
    duration = st.slider("Recording Duration (seconds)", min_value=2, max_value=15, value=5)

    if st.button("🎤 Start Live Speech"):
        # Record audio and get temporary file path
        audio_file = record_audio(duration=duration)
        
        # Transcribe speech with the selected language
        language_code = LANGUAGE_CODES.get(selected_language, "en-US")
        spoken_text = speech_to_text(audio_file, language=language_code)
        
        if "Speech not recognized" in spoken_text or "Error" in spoken_text:
            st.error(spoken_text)
        else:
            # Extract phonemes
            expected_phonemes = text_to_phonemes(reference_text, selected_language)
            spoken_phonemes = text_to_phonemes(spoken_text, selected_language)

            # Pronunciation accuracy and correction suggestions
            overall_accuracy, word_feedback = compare_phonemes(expected_phonemes, spoken_phonemes)

            # Fluency scoring
            fluency = calculate_fluency(reference_text, duration=duration)

            # Store scores for trend graph
            st.session_state.fluency_scores.append(fluency)
            st.session_state.pronunciation_accuracies.append(overall_accuracy)
            
            # Store last analysis results for chatbot
            st.session_state.last_word_feedback = word_feedback
            st.session_state.last_overall_accuracy = overall_accuracy
            st.session_state.last_fluency = fluency
            st.session_state.last_spoken_text = spoken_text
            st.session_state.last_reference_text = reference_text

            # Display results
            st.subheader("📑 Pronunciation Analysis")
            st.write(f"✅ **You Said:** {spoken_text}")
            st.write(f"🔠 **Overall Pronunciation Accuracy:** {overall_accuracy}%")
            st.write(f"⚡ **Fluency Score:** {round(fluency, 2)}%")

            # Generate and display chatbot response with language support
            chatbot_response = generate_chatbot_response(
                overall_accuracy, fluency, word_feedback, spoken_text, reference_text, selected_language
            )
            st.session_state.chat_history.append(("Assistant", chatbot_response))
            
            # Show Graphs side by side
            st.subheader("📊 Visualization")
            fig = plot_side_by_side_charts(fluency, overall_accuracy, word_feedback)
            st.pyplot(fig)
            
            # Show trend graph if multiple attempts exist - FIXED CALL TO TREND GRAPH
            if len(st.session_state.fluency_scores) > 1:
                st.subheader("📈 Progress Trends")
                trend_fig = plot_trend_graph(st.session_state.fluency_scores, st.session_state.pronunciation_accuracies)
                st.pyplot(trend_fig)  # Pass the figure directly, not trend_fig.figure

            # Word-by-word analysis with a structured table
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
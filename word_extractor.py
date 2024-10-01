import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr


def process_speech():
    r = sr.Recognizer()
    language = lang_var.get()
    
    with sr.Microphone() as source:
        status_label.config(text="Adjusting for ambient noise... Please wait.")
        r.adjust_for_ambient_noise(source) 
        status_label.config(text="Listening...")
        try:
            audio = r.listen(source)
            speech = r.recognize_google(audio, language=language)
            text_box.delete(1.0, tk.END) 
            text_box.insert(tk.END, speech)  
            status_label.config(text="Speech recognized successfully.")
        except sr.UnknownValueError:
            status_label.config(text="Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            status_label.config(text=f"Could not request results; {e}")
        except Exception as e:
            status_label.config(text=f"An unexpected error occurred: {e}")


def clear_text():
    text_box.delete(1.0, tk.END) 
    status_label.config(text="Text cleared.")


root = tk.Tk()
root.title("Speech to Text")


status_label = tk.Label(root, text="Press the button and speak")
status_label.pack(pady=10)

text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=10)


language_label = tk.Label(root, text="Select Language:")
language_label.pack(pady=10)


languages = {
    "English": "en",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Tamil": "ta"
}
lang_var = tk.StringVar(value="en")

language_menu = tk.OptionMenu(root, lang_var, *languages.values())
language_menu.pack(pady=10)

process_button = tk.Button(root, text="Start Recording", command=process_speech)
process_button.pack(pady=10)

clear_button = tk.Button(root, text="Clear", command=clear_text)
clear_button.pack(pady=10)

root.mainloop()


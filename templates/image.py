import requests
from PIL import Image
from io import BytesIO
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


nltk.download('punkt_tab')
nltk.download('stopwords')


GOOGLE_API_KEY = 'AIzaSyABUgegHP9oShOT9xtw2Ux6ehrZV3AXw70'  # Replace with your Google API Key
GOOGLE_CX = '1190d6ea6abc94b16'  # Replace with your Custom Search Engine ID
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


# Function to fetch image from Google Custom Search APImN
def fetch_image_from_google(word):
    params = {
        "q": word,
        "cx": GOOGLE_CX,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 1
    }
    
    response = requests.get(GOOGLE_SEARCH_URL, params=params)
    
    if response.status_code == 200:
        search_results = response.json()
        if 'items' in search_results and len(search_results['items']) > 0:
            image_url = search_results['items'][0]['link']
            
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                return Image.open(BytesIO(image_response.content))
    
    return None


# Function to display images in real time
def display_images_real_time(words):
    for word in words:
        image = fetch_image_from_google(word)
        
        if image is not None:
            image_np = np.array(image)
            
            plt.imshow(image_np)
            plt.title(word)
            plt.axis('off')  
            plt.show(block=False)
            
            plt.pause(2)  
            plt.clf()  
        else:
            print(f"No image found for word: {word}")
    
    plt.close('all')


# Function to process text and display corresponding images
def text_to_real_time_images(text):
    words = nltk.word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    if filtered_words:
        display_images_real_time(filtered_words)
    else:
        print("No meaningful words found for image generation.")


# Main function
if __name__ == "__main__":
    input_text = input("Enter a text: ")
    text_to_real_time_images(input_text)
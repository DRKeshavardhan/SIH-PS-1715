from googletrans import Translator
import json
import os
# from nltk.parse import stanford
import stanza 
import uuid
stanza.download('en',model_dir='stanza_resources')
# stanza.install_corenlp()
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
from six.moves import urllib
import zipfile
import sys
import time
import ssl
import speech_recognition as sr

ssl._create_default_https_context = ssl._create_unverified_context
from flask import Flask,request,render_template,send_from_directory,jsonify
translator = Translator()
app =Flask(__name__,static_folder='static', static_url_path='')

temp_files = set()
import stanza
import pprint 

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Download zip file from https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip and extract in stanford-parser-full-2015-04-20 folder in higher directory
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR,
                                             'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'


def is_parser_jar_file_present():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    return os.path.exists(stanford_parser_zip_file_path)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.perf_counter()
        return
    duration = time.perf_counter() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    url = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
    urllib.request.urlretrieve(url, stanford_parser_zip_file_path, reporthook)

def extract_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    try:
        with zipfile.ZipFile(stanford_parser_zip_file_path) as z:
            z.extractall(path=BASE_DIR)
    except Exception:
        os.remove(stanford_parser_zip_file_path)
        download_parser_jar_file()
        extract_parser_jar_file()

def extract_models_jar_file():
    stanford_models_zip_file_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
    stanford_models_dir = os.environ.get('CLASSPATH')
    with zipfile.ZipFile(stanford_models_zip_file_path) as z:
        z.extractall(path=stanford_models_dir)


def download_required_packages():
    if not os.path.exists(os.environ.get('CLASSPATH')):
        if is_parser_jar_file_present():
           pass
        else:
            download_parser_jar_file()
        extract_parser_jar_file()

    if not os.path.exists(os.environ.get('STANFORD_MODELS')):
        extract_models_jar_file()


en_nlp = stanza.Pipeline('en',processors={'tokenize':'spacy'})	


stop_words = set(["am","are","is","was","were","be","being","been","have","has","had",
					"does","did","could","should","would","can","shall","will","may","might","must","let"]);



sent_list = []
sent_list_detailed=[]


word_list=[]

word_list_detailed=[]

def convert_to_sentence_list(text):
	for sentence in text.sentences:
		sent_list.append(sentence.text)
		sent_list_detailed.append(sentence)

def convert_to_word_list(sentences):
	temp_list=[]
	temp_list_detailed=[]
	for sentence in sentences:
		for word in sentence.words:
			temp_list.append(word.text)
			temp_list_detailed.append(word)
		word_list.append(temp_list.copy())
		word_list_detailed.append(temp_list_detailed.copy())
		temp_list.clear()
		temp_list_detailed.clear()

def filter_words(word_list):
	temp_list=[]
	final_words=[]
	
	for words in word_list:
		temp_list.clear();
		for word in words:
			if word not in stop_words:
				temp_list.append(word);
		final_words.append(temp_list.copy());
	
	for words in word_list_detailed:
		for i,word in enumerate(words):
			if(words[i].text in stop_words):
				del words[i]
				break
	
	return final_words

def remove_punct(word_list):
	
	for words,words_detailed in zip(word_list,word_list_detailed):
		for i,(word,word_detailed) in enumerate(zip(words,words_detailed)):
			if(word_detailed.upos=='PUNCT'):
				del words_detailed[i];
				words.remove(word_detailed.text);
				break;


# lemmatizes words
def lemmatize(final_word_list):
	for words,final in zip(word_list_detailed,final_word_list):
		for i,(word,fin) in enumerate(zip(words,final)):
			if fin in word.text:
				if(len(fin)==1):
					final[i]=fin;
				else:
					final[i]=word.lemma;
				
	
	for word in final_word_list:
		print("final_words",word);

def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}

    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag

def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i = i + 1
    return i, modified_parse_tree


def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i = i + 1
    return i, modified_parse_tree


# modifies the tree according to POS
def modify_tree_structure(parent_tree):
    # Mark all subtrees position as 0
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    # Initialize new parse tree
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    # recursively check for omitted clauses to be inserted in tree
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:  #check if subtree leads to some word
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i = i + 1

    return modified_parse_tree

def reorder_eng_to_isl(input_string):
	download_required_packages();
	# check if all the words entered are alphabets.
	count=0
	for word in input_string:
		if(len(word)==1):
			count+=1;

	if(count==len(input_string)):
		return input_string;
	
	parser = StanfordParser()
	# Generates all possible parse trees sort by probability for the sentence
	possible_parse_tree_list = [tree for tree in parser.parse(input_string)]
	print("i am testing this",possible_parse_tree_list)
	# Get most probable parse tree
	parse_tree = possible_parse_tree_list[0]
	# print(parse_tree)
	# Convert into tree data structure
	parent_tree = ParentedTree.convert(parse_tree)
	
	modified_parse_tree = modify_tree_structure(parent_tree)
	
	parsed_sent = modified_parse_tree.leaves()
	return parsed_sent


# final word list
final_words= [];
# final word list that is detailed(dict)
final_words_detailed=[];


# pre processing text
def pre_process(text):
	remove_punct(word_list)
	final_words.extend(filter_words(word_list));
	lemmatize(final_words)

# checks if sigml file exists of the word if not use letters for the words
def final_output(input):
	final_string=""
	valid_words=open("words.txt",'r').read();
	valid_words=valid_words.split('\n')
	fin_words=[]
	for word in input:
		word=word.lower()
		if(word not in valid_words):
			for letter in word:
				# final_string+=" "+letter
				fin_words.append(letter);
		else:
			fin_words.append(word);

	return fin_words

final_output_in_sent=[]

# converts the final list of words in a final list with letters seperated if needed
def convert_to_final():
	for words in final_words:
		final_output_in_sent.append(final_output(words))

# takes input from the user
def take_input(text):
	test_input=text.strip().replace("\n","").replace("\t","")
	test_input2=""
	if(len(test_input)==1):
		test_input2=test_input;
	else:
		for word in test_input.split("."):
			test_input2+= word.capitalize()+" .";

	# pass the text through stanza
	some_text= en_nlp(test_input2);
	convert(some_text);


def convert(some_text):
	convert_to_sentence_list(some_text);
	convert_to_word_list(sent_list_detailed)

	# reorders the words in input
	for i,words in enumerate(word_list):
		word_list[i]=reorder_eng_to_isl(words)

	# removes punctuation and lemmatizes words
	pre_process(some_text);
	convert_to_final();
	remove_punct(final_output_in_sent)
	print_lists();

def print_lists():
	print("--------------------Word List------------------------");
	pprint.pprint(word_list)
	print("--------------------Final Words------------------------");
	pprint.pprint(final_words)
	print("---------------Final sentence with letters--------------")
	pprint.pprint(final_output_in_sent)

# clears all the list after completing the work
def clear_all():
	sent_list.clear()
	sent_list_detailed.clear()
	word_list.clear()
	word_list_detailed.clear()
	final_words.clear()
	final_words_detailed.clear()
	final_output_in_sent.clear()
	final_words_dict.clear()


final_words_dict = {}

@app.route('/',methods=['GET'])
def index():
	clear_all()
	return render_template('index.html')


@app.route('/',methods=['GET','POST'])
def flask_test():
	clear_all();
	
	text = request.form.get('text') #gets the text data from input field of front end
	print("text is", text)
	if(text==""):
		return ""
	take_input(text)
 
	for words in final_output_in_sent:
		for i,word in enumerate(words,start=1):
			final_words_dict[i]=word

	print("---------------Final words dict--------------");

	for key in final_words_dict.keys():
		if len(final_words_dict[key])==1:
			final_words_dict[key]=final_words_dict[key].upper()
	print(final_words_dict)
	temp_filename = f"temp_{uuid.uuid4()}.sigml"
	temp_filepath = os.path.join('static/SignFiles', temp_filename)
    
	temp_files.add(temp_filepath)
	with open(temp_filepath,'a') as temp:
		temp.write("<sigml>\n")
		# with open(temp_filepath,'a') as temp:
		# 		temp.write("<transition gloss='transition'><hammovecont/></transition>")
	for key,value in final_words_dict.items():
		path=os.path.join('static/SignFiles',value+'.sigml')
		with open(path,'r') as f:
			lines=f.readlines()
			with open(temp_filepath,'a') as temp:
				for line in lines[1:-1]:
					temp.write(line)
			with open(temp_filepath,'a') as temp:
				temp.write("<transition gloss='transition'><hammovecont/></transition>")
		print(path)
	with open(temp_filepath,'a') as temp:
		temp.write("</sigml>")
	dic={}
	dic[1]=f'{temp_filename}'
	# index=1
	# for key,value in final_words_dict.items():
	# 	d[index]=value
	# 	if len(final_words_dict)-1!=index:
	# 		d[index+1]="transition"
	# 	index+=2
	print('hello everyone')
	return dic

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    r = sr.Recognizer()
    language = request.json.get('language', 'en')
    print(f"Language: {language}")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source)
            speech = r.recognize_google(audio, language=language)
            print(f"Recognized Speech: {speech}") 
            if language!='en':
                speech = translator.translate(speech, dest='en').text
                print(speech)
            return jsonify({"success": True, "speech": speech})
        except sr.UnknownValueError:
            return jsonify({"success": False, "error": "Could not understand the audio"})
        except sr.RequestError as e:
            return jsonify({"success": False, "error": f"Request failed; {e}"})
        except Exception as e:
            return jsonify({"success": False, "error": f"An unexpected error occurred: {e}"})


@app.route('/static/<path:path>')
def serve_signfiles(path):
	print("here")
	print(path)
	return send_from_directory('static',path)

if __name__=="__main__": 
    app.run(host='0.0.0.0')

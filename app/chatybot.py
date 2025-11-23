import os
import json
import spacy
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim


# Define the Neural Network model architecture
class LLM(nn.Module):
    def __init__(self, input_size, output_size):
        super(LLM, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 125)
        self.fc3 = nn.Linear(125, output_size) # Output layer size matches the number of intents
        self.relu = nn.ReLU() # the ReLU activation function
        self.dropout = nn.Dropout(0.5) # the dropout layer to prevent overfitting

    def forward(self, x):
        # Define the forward pass of the network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        # No activation function on the last layer because CrossEntropyLoss will be used
        return out
    

    
import nltk
# Download necessary NLTK data files. 'punkt' is for tokenization, 'wordnet' is for lemmatization.
nltk.download('punkt_tab')
nltk.download('wordnet')
    
# Main class for the chatbot logic, handling data processing, training, and prediction.
class TextProcessor:
    
    def __init__(self, intents_path, function_mappings=None):
        # Initialize instance variables
        self.model = None # To hold the trained PyTorch model
        self.documents  = [] # To store patterns and their associated tags
        self.intents_path = intents_path # Path to the JSON intents file
        self.intents = [] # A list of all unique intent tags
        self.intents_responses = {} # A dictionary mapping tags to their possible responses
        self.vocabulary = [] # A list of all unique words in the patterns (for bag-of-words)
        # A dictionary mapping intent tags to functions that should be executed
        self.function_mappings = function_mappings if function_mappings else {}
        
        self.X = None # To store the training data (features)
        self.y = None # To store the training data (labels)

    @staticmethod
    def tokenize_and_lemmatize(sentence):
        # Initialize the WordNet lemmatizer
        lemmatizer = nltk.WordNetLemmatizer()
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        # Lemmatize each word to its base form and convert to lowercase
        return [lemmatizer.lemmatize(word.lower()) for word in words]


    @staticmethod
    def stem(word):
        # Initialize the Porter stemmer
        stemmer = nltk.PorterStemmer()
        # Stem the word to its root form and convert to lowercase
        return stemmer.stem(word.lower())

    @staticmethod
    def bag_of_words(tokenized_sentence, words):
        # Stem each word in the input sentence
        tokenized_sentence = [TextProcessor.stem(word) for word in tokenized_sentence]
        # Create a bag (vector) of zeros with the length of the vocabulary
        bag = np.zeros(len(words), dtype=np.float32)
        # Iterate through the vocabulary
        for idx, w in enumerate(words):
            # If a word from the vocabulary is in the sentence, set the corresponding bag position to 1
            if w in tokenized_sentence:
                bag[idx] = 1
        return bag

    def load_intents(self):
        # Check if the intents file exists
        if os.path.exists(self.intents_path):
            # Open and load the JSON data from the file
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            # Iterate through each intent in the loaded data
            for intent in intents_data['intents']:
                # If the intent tag is new, add it to the list of intents and store its responses
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent.get('responses', [])
                    
                # For each pattern in the intent
                for pattern in intent['patterns']:
                    # Tokenize and lemmatize the pattern
                    tokenized_pattern = TextProcessor.tokenize_and_lemmatize(pattern)
                    # Add the tokenized pattern and its tag to the documents list
                    self.documents.append((tokenized_pattern, intent['tag']))
                    # Add the new words to the vocabulary
                    self.vocabulary.extend(tokenized_pattern)

            # Create a sorted list of unique words for the vocabulary
            self.vocabulary = sorted(list(set(self.vocabulary)))
        else:
            # Raise an error if the intents file is not found
            raise FileNotFoundError(f"Intents file not found at path: {self.intents_path}")


    def create_training_data(self):
        # Load the intents from the JSON file first
        self.load_intents()
        bags = [] # List to hold the bag-of-words vectors
        indices = [] # List to hold the numerical labels (indices of intents)

        # Iterate through each document (pattern sentence and its tag)
        for (pattern_sentence, tag) in self.documents:
            # Create a bag-of-words vector for the pattern
            bag = TextProcessor.bag_of_words(pattern_sentence, self.vocabulary)
            bags.append(bag)
            # Append the index of the tag as the label
            indices.append(self.intents.index(tag))

        # Convert the lists to NumPy arrays for training
        self.X = np.array(bags)
        self.y = np.array(indices)
###################################################################################################################
    def training_model(self, input_size, output_size, hidden_size=8, num_epochs=1000, batch_size=8, learning_rate=0.001):
        # Convert training data from NumPy arrays to PyTorch tensors
        X_tensor = torch.tensor(self.X).float()
        y_tensor = torch.tensor(self.y).long()

        # Create a TensorDataset and a DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network model
        model = LLM(input_size, output_size)
        # Define the loss function (CrossEntropyLoss is suitable for multi-class classification)
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer (Adam is a popular choice)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Start the training loop
        for epoch in range(num_epochs):
            # Iterate over the data in batches
            for i, (inputs, labels) in enumerate(dataloader):
                # Zero the gradients to prevent accumulation
                optimizer.zero_grad()
                # Forward pass: get model predictions
                outputs = model(inputs)
                # Calculate the loss
                loss = criterion(outputs, labels)
                # Backward pass: compute gradients
                loss.backward()
                # Update the model's weights
                optimizer.step()

            # Print the loss every 100 epochs for monitoring
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Store the trained model in the instance
        self.model = model

    def save_model(self, model_path):
        # Check if a model has been trained
        self.model = None
        if self.model:
            # Save the model's state dictionary to the specified path
            torch.save(self.model.state_dict(), model_path)
        else:
            # Raise an error if there is no model to save
            raise ValueError("Model has not been trained yet.")
            
    def load_model(self, model_path, input_size, output_size):
        # Initialize a new model with the same architecture
        model = LLM(input_size, output_size)
        # Load the saved weights and biases into the model
        model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode (disables dropout, etc.)
        model.eval()
        # Store the loaded model in the instance
        self.model = model
        
    def predict_intent(self, sentence):
        # Ensure a model is loaded or trained
        if not self.model:
            raise ValueError("Model is not loaded. Please load or train the model first.")

        # Preprocess the input sentence in the same way as the training data
        tokenized_sentence = TextProcessor.tokenize_and_lemmatize(sentence)
        bag = TextProcessor.bag_of_words(tokenized_sentence, self.vocabulary)
        # Convert the bag-of-words array to a PyTorch tensor and add a batch dimension
        input_tensor = torch.tensor(bag).float().unsqueeze(0)

        # Disable gradient calculation for inference
        with torch.no_grad():
            # Get the model's raw output (logits)
            output = self.model(input_tensor)
            # Find the index of the highest score, which corresponds to the predicted intent
            _, predicted = torch.max(output, dim=1)
            intent_index = predicted.item()
            # Get the string tag for the predicted index
            intent_tag = self.intents[intent_index]

        return intent_tag   
    
    def process_message(self, message):
        # Predict the intent of the user's message
        intent_tag = self.predict_intent(message)
        # Get the list of possible responses for the predicted intent
        responses = self.intents_responses.get(intent_tag, [])
        # Choose a random response from the list, or provide a default if no responses are found
        response = random.choice(responses) if responses else "I'm not sure how to respond to that."
        
        # Check if the predicted intent is associated with a function to be executed
        if intent_tag in self.function_mappings:
            # Get the function from the mappings
            func = self.function_mappings[intent_tag]
            # Execute the function and get its response
            func_response = func()
            # Append the function's response to the main response
            response += f" {func_response}"
        
        return response

# This block runs only when the script is executed directly (not when imported)
if __name__ == '__main__':
    # Initialize a dictionary to hold the combined intents from all files
    all_intents = {'intents': []}
    # List of paths to the intent JSON files
    intent_files = [
        '/home/ngobeni/Documents/Programming/project/clinicM/data/general_chat.json',
        '/home/ngobeni/Documents/Programming/project/clinicM/data/task_automation.json',
        '/home/ngobeni/Documents/Programming/project/clinicM/data/medical_triage.json'
    ]

    # Loop through each file path
    for file_path in intent_files:
        # Open and load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extend the 'intents' list in the main dictionary with the intents from the current file
            all_intents['intents'].extend(data['intents'])

    # Define the path for the new combined intents file
    combined_intents_path = '/home/ngobeni/Documents/Programming/project/clinicM/data/combined_intents.json'
    # Write the combined intents dictionary to a new JSON file
    with open(combined_intents_path, 'w') as f:
        json.dump(all_intents, f, indent=2)

    # Initialize the TextProcessor with the path to the combined intents file
    processor = TextProcessor(intents_path=combined_intents_path)
    # Create the training data (bag-of-words vectors and labels)
    processor.create_training_data()

    # Determine the input size from the length of a bag-of-words vector (vocabulary size)
    input_size = len(processor.X[0])
    # Determine the output size from the number of unique intents
    output_size = len(processor.intents)

    # Train the neural network model
    processor.training_model(input_size=input_size, output_size=output_size)

    # Inform the user that the bot is ready for conversation
    print("Bot is ready! Type 'quit or q' to exit.")

    # Start an infinite loop for the conversation
    while True:
        # Get input from the user
        user_input = input("You: ")
        # If the user types 'quit', break the loop to end the program
        if user_input.lower() == 'quit' or user_input.lower() == 'q':
            break
        # Process the user's message to get a response from the bot
        response = processor.process_message(user_input)
        # Print the bot's response
        print(f"Bot: {response}")
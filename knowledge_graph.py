import stanza

# Load the English model in Stanza
nlp = stanza.Pipeline('en')

# Define the input text (story)
text = "Unlike the others in his troop, Tiko was always curious about the unknown . Tiko realized that knowledge wasn’t just about seeing—it was about understanding . I'm looking for something special, Tiko replied . The Jungle’s Secret Deep in the heart of the jungle, where the trees whispered secrets and the rivers sang old songs, a young monkey named Tiko swung from vine to vine . One day, he noticed an old parrot named Azul perched on a twisted tree, watching him closely . The sky was green, the leaves were blue, and in the reflection, Tiko saw animals he had never met before . Azul croaked . Azul fluffed his feathers . Instead, he shared the wisdom of the jungle, teaching others to listen, observe, and appreciate the mysteries hidden in plain sight ."

# Process the text with the NLP pipeline
doc = nlp(text)

# Extract entities and relationships
entities = []
relationships = []

# Extract entities from the document
for ent in doc.ents:
    entities.append((ent.text, ent.type))

# Extract subject-verb-object (or subject-attribute) triples from the sentences
for sent in doc.sentences:
    # Initialize for each sentence
    subject = None
    verb = None
    obj = None

    # Loop through the words in the sentence
    for word in sent.words:
        # Check if the word is a subject (nsubj or nsubj:pass)
        if 'nsubj' in word.deprel:
            subject = word.text
        # Check if the word is a verb (root or verb-related dependencies)
        elif word.upos == 'VERB':
            verb = word.text
        # Check if the word is an object (obj, iobj, or obl)
        elif word.deprel in ['obj', 'iobj', 'obl']:
            obj = word.text

        # If we have all three components, add the relationship
        if subject and verb and obj:
            relationships.append((subject, verb, obj))
            # Reset only the object after adding the relationship to allow for multiple objects
            obj = None

    # Clear subject and verb after processing each sentence
    subject = None
    verb = None

# Print the extracted entities and relationships
print('Entities:', entities)
print('Relationships:', relationships)

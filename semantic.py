import spacy

nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# the cat and the monkey score higher on the similarity because they are both
#the cat and the monkey have higher score when it comes to simularity because they are both animals
#also we normally associate a monkey with a banana because monkeys like them 
#compared to a cat and banana

word4 = nlp("car")
word5 = nlp("airplane")
word6 = nlp("truck")

print(word4.similarity(word5))
print(word6.similarity(word5))
print(word6.similarity(word4))

#a truck and a car a simulare because the are land vehicles 
# as compared to planes which flies in the air 
#the simularity between all of them is that they all have engines 



tokens = nlp ("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# apples and bananas are both fruits which makes then simular
# monkey and cat are animals.  Also monkey and banana is similar compared to monkey and apple, and cat and apple
# cat  and  banana because of how we associate bananas with monkeys

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#the word car is mentioned more than any other name
# reading these sentences you realise that this person is speaking about themselves and the things they own
# the first sentence and last sentece are simular as they as speaking about the dog
#the second sentence and the last sentence are simular as they speak of the persons car

# a note on what you notice is different from the model'en_core_web_md 

#The en_core_web_md model is a spaCy model for natural language processing that is trained on web data and includes the full vocabulary and word vectors.
# It is a larger and more versatile model than the en_core_web_sm, which is a smaller model that includes only the most common words and word vectors.
# en_core_web_md is more useful for more complex NLP tasks, such as entity recognition and text classification, while en_core_web_sm is more appropriate for simpler tasks,
# such as tokenization and lemmatization. In other words, the 'md' in en_core_web_md stands for 'medium' dataset and 'sm' in en_core_web_sm stands for 'small' dataset,
# So en_core_web_md contains more data and therefore can perform tasks with more accuracy than en_core_web_sm
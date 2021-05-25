# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import numpy as np
import collections
from porter_stemmer import PorterStemmer
from collections import deque
import random

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""
    def __init__(self, creative=False):
        self.name = 'Chris'
        self.creative = creative
        self.flipWords = self.stem(["not","didn't", "never", "wasn't"], True)
        self.fillerWords = self.stem(["really","absolutely", "undoubtedbly", "honestly"], True)
        self.titlepattern = "\"([\w\.'é:\-\&/!?ó*\[\]\(\)\[, \]]+)\""
        self.userData = []
        self.certain = False # make sure this is true before moving on
        self.go_back_mode = "newMovie"
        self.archive = (None, None)
        self.archiveList = []
        self.recommendations = None
        self.silentGame = 0
        self.titleOptionsToNarrow = None
        self.mode = "newMovie" # "ask" "confirm" "clarifySentiment" "predict"
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        
        # Dictionary Mapping Alternate Titles to movie index
        self.alternate_titles = {}
        
        # Creative Disambiguation (part 1) Dictionary Mapping Each title token to movie index
        self.disamb_title = collections.defaultdict(list)
        
        self.titleDict = self.refineTitles(self.titles)
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.newsentiment = self.refineSentiment(self.sentiment)

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hey, I'm Chris! I'm going to recommend a movie to you, but first, I will ask you about your taste in movies. Tell me about a movie that you've seen!"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################
        self.go_back_mode = "end"
        goodbye_message = "I'm glad i could help!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
        
    def makePrediction(self, prefix, creative = False):
        """
        make a prediction based off the data we have
        """
        ########################################################################
        # TODO: Edit for creative                    #
        ########################################################################

        if len(self.userData) >= 5:
            if self.mode != "predict":
                self.mode = "predict"
                user_ratings = np.zeros(len(self.ratings))
                for index, sentiment in self.userData:
                    user_ratings[index] = sentiment
                self.recommendations = deque(list(self.recommend(user_ratings, self.ratings,len(self.ratings) - 5 , creative)))
                recommendation = self.titles[self.recommendations[0]][0]
                self.recommendations.popleft()
                prompt =  "{} From what you've told me, I think you might like {}.\n do you want another recommendation?"\
                .format(prefix, recommendation)
                self.go_back_mode = "predict"
                return self.confirmResponse(prompt, "")
            else:
                if(not self.recommendations):
                    self.mode = "end"
                    return "I seem to have run out of recommendations sorry"
                recommendation = self.titles[self.recommendations[0]][0]
                self.recommendations.popleft()
                notRobot = ["you would also like {}, should I keep going?".format(recommendation),
                            "{} would be an perfect movie for you, want another?".format(recommendation),
                            "you could try watching {}, you want to see what else I got?".format(recommendation),
                            "My best guess is that {} will make for a perfect movie for you. Want to see my next best?".format(recommendation)]
                self.go_back_mode = "predict"
                return self.confirmResponse(random.choice(notRobot), "")
        elif self.archive[1] == 0 or not self.certain:
            return self.process("Dummy data: {}".format(self.archive[0]))
        else:
            self.certain = False
            self.mode = "newMovie"
            self.archive = (None, None)
            self.archiveList = []
            return "{} So tell me more about another movie you've seen".format(prefix)

        ########################################################################

    def confirmResponse(self, prompt, user_input = ""):
        """
        confirm wether the sentiment and movie we detect is correct or not
        """
        ########################################################################
        # TODO: Edit for creative                    #
        ########################################################################

        if self.mode != "confirm":
            self.mode = "confirm"
            return prompt
        yorn = user_input.replace(" ","").replace(".","").replace("!","")
        affirmitives = ["y","yeah","yes","mhm","yup","ye","yay","yea"]
        nonAffirmitives = ["n","nope","no","mm","nah","nada","nay"]
        if yorn in affirmitives:
            if self.archive != None: 
                self.userData.append(self.archive)
            if len(self.archiveList) > 0 : 
                self.certain = True
                for archive in self.archiveList:
                    self.userData.append(archive)
            self.mode = self.go_back_mode
            return self.makePrediction("Ok cool!")
        elif yorn in nonAffirmitives:
            # hacky way to end
            if(self.go_back_mode == "predict"):
                return self.goodbye()
            self.mode = "newMovie"
            self.certain = False
            notRobot = ["oh, ok sorry for misunderstanding, what other movie have you seen?", 
                        "Sorry im just brain freezing right now, what other movie have you watched", 
                        "Yeah, lets just skip this one, tell me about anoter movie",
                        "Lets talk about another movie"]
            return  random.choice(notRobot)
        else:
            return 'I didn\'t get that, try answering "yes" or "no"'

        ########################################################################
    
    def clarifySentiment(self, user_input):
        """
        confirm wether the sentiment and movie we detect is correct or not
        """
        ########################################################################
        # TODO: Edit for creative                    #
        ########################################################################
        if self.mode !=  "clarifySentiment":
            self.mode = "clarifySentiment"
            movie = self.titles[self.archive[0]][0]
            notRobot = ["Here's the thing, i dont know if you like {} or not can you tell me more?".format(movie), 
                        "So.... do you like {} or not, talk to me".format(movie)]
            return random.choice(notRobot)
        else:
            sentiment = self.extract_sentiment(user_input)
            if(sentiment == 0): 
                self.mode = "" # clear mode so bot can ask question again, recurse
                return self.clarifySentiment("")
            self.archive = (self.archive[0], sentiment)
            #self.go_back_mode = "newMovie"
            return self.confirmResponse("so you hate it" if sentiment == -1 else "so you love it")

        ########################################################################
    
    def clarifyTitle(self, user_input, possible_titles):
        """
        confirm which title when there are more than 1 options
        """
        ########################################################################
        # TODO: Edit for creative                    #
        ########################################################################
        if not self.creative:
            self.mode = "newMovie"
            return "there is more than one title that matches {} please specify".format(user_input)
        else:
            if self.mode != "clarifyTitle":
                self.mode = "clarifyTitle"
                response = "Which one did you mean?\n"
                self.titleOptionsToNarrow = possible_titles
                for i, title in enumerate(possible_titles):
                    response += "{}. {}\n".format(i + 1,self.titles[title][0])
                return response
            else:
                self.titleOptionsToNarrow = self.disambiguate(user_input, self.titleOptionsToNarrow)
                if len(self.titleOptionsToNarrow) == 1:
                    self.mode = "confirmArchive"
                    self.archive = (self.titleOptionsToNarrow[0],self.archive[1])
                    return self.process("random stuff that doesnt matter")
                else:
                    response = "try to get more specific, these options are still a possibility\n"
                    for i, title in enumerate(self.titleOptionsToNarrow):
                        response += "{}. {}\n".format(i + 1,self.titles[title][0])
                    return response
            return "there is more than one title that matches {} please specify".format(user_input)

        ########################################################################
    def processUserTitle(self, user_input_title):
        """
        find all the matches, narrrow down to one, update archive
        """
        ########################################################################
        # TODO: Edit for creative                    #
        ########################################################################
        movie_ids = self.find_movies_by_title(user_input_title)
        if len(movie_ids) == 0 and not self.creative: 
            notRobot = ["Hm, I haven't seen that movie. Tell me about another one.", 
                            "Haven't heard of {} try another movie".format(user_input_title)]
            return random.choice(notRobot)

        elif len(movie_ids) == 0 and self.creative:
            movie_ids = self.find_movies_closest_to_title(user_input_title)
            if len(movie_ids) == 0:
                notRobot = ["Hm, I haven't seen that movie. Tell me about another one.", 
                            "Haven't heard of {} try another movie".format(user_input_title), 
                            "Ooooooo I love that movie!....sike, talk about another one",
                            "I bet {} would be great to talk about but can you tell me about a different one?".format(user_input_title)]
                return random.choice(notRobot)
            if len(movie_ids) == 1:
                self.archive = (movie_ids[0], self.archive[1])
                movie = self.titles[movie_ids[0]][0]
                notRobot = ["Do you mean {}".format(movie), 
                            "I've heard of {} is that what you're talking about?".format(movie),
                            "Is {} right?".format(movie)]
                self.go_back_mode = "confirmArchive"
                return self.confirmResponse(random.choice(notRobot))
        elif len(movie_ids) == 1:
            self.mode = "confirmArchive"
            self.archive = (movie_ids[0], self.archive[1])
            return self.process("Nothing matters")
        elif len(movie_ids) > 1:
            return self.clarifyTitle(user_input_title, movie_ids)

        ########################################################################

    def handleMultipleTitles(self, user_input):
        """
        get more than one sentiment at a time
        """
        senti = {-1: "don't like", 0: "are indifferent to", 1: "like"}
        results = self.extract_sentiment_for_movies(user_input)
        for result in results:
            title_indices = self.find_movies_by_title(result[0])
            if(len(title_indices) < 1):
                self.archiveList = []
                self.archiveList = (None, None)
                self.mode = "newMovie"
                return "couldn't find one of those titles, lets try another movie"
            self.archive = (title_indices[0],result[1])
            self.archiveList.append((title_indices[0],result[1]))
        prompt = "So you {} {}".format(senti[results[0][1]],results[0][0])
        for i in range(1, len(result)):
            prompt += " and {} {}".format(senti[results[i][1]],results[i][0])
        prompt += ", is that correct?"
        self.go_back_mode = "newMovie"
        return self.confirmResponse(prompt)
        
    def handleEmpty(self, user_input):
        """
        this function is completely useless cuz they handle empty strings for you already
        """
        if(self.silentGame > 0):
                self.silentGame = self.silentGame - 1
                return "                           "
        if user_input.replace(".", "").replace(",", "").replace(":", "") == "":
            notRobot = ["Please type something"]
            if self.creative:
                notRobot = ["Say something to me hun", "Why dont you wanna talk to me :(", "Oh wow, the silent treatment", 
                            "two can play at that game", "....um is your keyboard working", "stop it, be reasonable",
                            "I need words dude, actual words", "Sigh, please say words", "Okay i'm getting upset" ]
            choice = random.choice(notRobot)
            if choice == "two can play at that game":
                self.silentGame = 3
            return choice
        else: return None
    
    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        
        # if self.creative:
        #     preprocessed_line = self.preprocess(line)
        #     potential_movies = self.extract_titles(preprocessed_line)
        #     if len(potential_movies) == 0: return "Sorry, I don't understand. Tell me about a movie that you have seen." 
            
        #     # If the bot didn't find any movies ask again
        #     movie_ids = self.find_movies_by_title(potential_movies[0])
        #     movie_options = []
            
        #     # Get the movie titles 
        #     for movie_id in movie_ids:
        #         movie_options.append(self.titles[movie_id][0])
        #     if len(movie_options) == 0: return "Sorry, I couldn't find that movie. Tell me about a movie that you have seen."     

        # method variables
        preprocessed_line = self.preprocess(line)
        potential_movies = None
    
        # handle empty strings
        res = self.handleEmpty(preprocessed_line)
        if(res != None): return res
        
        if self.mode == "end":
            print("self.mode == end")
            return self.goodbye()
        if self.mode == "predict":
            print("self.mode == predict")
            return self.makePrediction("")
        if self.mode == "confirm":
            print("self.mode == confirm")
            return self.confirmResponse("", preprocessed_line)
        if self.mode == "clarifySentiment":
            print("self.mode == clarifySentiment")
            return self.clarifySentiment(preprocessed_line)
        if self.mode == "clarifyTitle":
            print("self.mode == clarifyTitle")
            return self.clarifyTitle(preprocessed_line, None)
        if self.mode == "handleMultipleTitles":
            print("self.mode == handleMultipleTitles")
            return self.handleMultipleTitles(None)
        if self.mode == "newMovie":
            print("self.mode == newMovie")
            sentiment = self.extract_sentiment(preprocessed_line)
            self.archive = (self.archive[0], sentiment)
            potential_movies = self.extract_titles(preprocessed_line)
            # check if user put at least 1 movie in quotes 
            if len(potential_movies) == 0: 
                return "Sorry, I don't understand. Try putting movies in quotes." 
            if len(potential_movies) == 1: 
                return self.processUserTitle(potential_movies[0])
            if len(potential_movies) > 1:
                return self.handleMultipleTitles(preprocessed_line) 
            

        if self.mode == "confirmArchive":
            print("self.mode == confirmArchive")
            print("here")
            if self.archive[1] == 0:
                self.go_back_mode = "confirmArchive"
                return self.clarifySentiment("")
            movie_id, sentiment = self.archive
            sentiment_in_english = "really liked" if sentiment == 1 else "weren't a big fan of"
            self.go_back_mode = "predict"
            self.certain = True
            return self.confirmResponse("Ok, sounds to me like you {} {}, that right?".format(sentiment_in_english, self.titles[movie_id][0]))
        self.certain = False
        self.mode = "newMovie"
        return "Uh... Lets talk about a new movie please"
    

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        
    
    def stem(self, x, islist = False):
        stemmer = PorterStemmer()
        if islist:
            return [self.stem(y) for y in x]
        return stemmer.stem(x, 0, len(x) - 1)
    
    def refineSentiment(self, sentiment):
        newSentiment = collections.defaultdict(int)
        
        for key, value in sentiment.items():
            newSentiment[self.stem(key)] = 1 if value == "pos" else -1
        return newSentiment

    def refineTitles(self, titleList):
        """Put self.titles in a better format

        Creative (Alternate/foreign titles) Finds the alternate titles 
        and adjusts for article in the alternate titles, 
        (Alternate Title is in parentheses)

        :param titleList: self.titles returned from util.load_ratings('data/ratings.txt')
        :returns: A dictionary with a key of the movie title and value of a list 
                    of the form [index, year, genre, article, alternativeTitle]
        """
        titles = collections.defaultdict(list)
        pattern = "((?:[\w\.'é:\+\-\&/!?ó*\[\]]+\s?)+)(?:\s|,\s(.+)\s)?(?:\((.+)\)\s)?(?:\((\d\d\d\d)-?(?:\d\d\d\d)?\))"

        for i in range(len(titleList)):
            titlefromlist, genre = titleList[i]
            res = re.findall(pattern, titlefromlist)
            if len(res) > 0:
                title, article, altTitle, year = re.findall(pattern, titlefromlist)[0]
                # Creative Disambiguation: index tokens of movie title to movie index
                # If there is an alt title, stores the alt title as key and movie index as value
                title_tokens = title.split()
                for token in title_tokens:
                    # Cleans the movie string tokens (i.e jackson: jackson)
                    processed_token = token.lower().replace(":","").replace(",","").replace(".","")
                    if processed_token in self.disamb_title:
                        curr_movies = self.disamb_title[processed_token]
                        curr_movies.append(i)
                        self.disamb_title[processed_token] = curr_movies
                    else:
                        self.disamb_title[processed_token] = [i]
                title = title.replace(" ", "").lower()
                titles[title].append([i, year, genre, article, altTitle])
                
                match = None
                # Checks if alternative title exsits
                # Creative adds article for alternate titles
                if len(altTitle) != 0:
                    try:
                        # Pattern to detect potenial articles such as Guerre du feu, 
                        # La and replace them to be LaGuerre du feu to index them in 
                        # the alternative titles dictionary which maps to movie index
                        alt_pattern = "((?:[\w\.'ôûé:\+\-\&!?ó*\[\]]+\s?)+)(?:, )?(\w+)?"        
                        match = re.findall(alt_pattern, altTitle)
                        first = re.findall(alt_pattern, altTitle)[0][0]
                        second = re.findall(alt_pattern, altTitle)[0][1]
                        altTitle = second + first
                        clean_alt_title  = altTitle.replace('a.k.a. ', "").lower().replace(" ", "")
                    except:
                        print("error", altTitle, match)
                    self.alternate_titles[clean_alt_title] = i
            # if title doesnt match regular expression, theres something really weird going on
            else:
                titles[titlefromlist].append([i, None, genre, None, None])
        return titles

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: it's just the text returned.                                   #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        # Added[, \] to adjust for haine, la
        pattern = "\"([\w\.'é:\-\&/!?ó*\[\]\(\)\[, \]]+)\""
        movie_options =  re.findall(pattern, preprocessed_input)
        
        return movie_options
    
    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        matches = []
        pattern = "(?:(?:([Aa]|[Tt]he|[Aa]n)) )?((?:(?:[\w.'é:\[, \]\-\&/!?ó]+) ?)+)(?:\((\d\d\d\d)\))?"
        article, u_title, year = re.findall(pattern, title)[0]
        # Preprocess the identified title
        u_title = u_title.replace(" ", "").lower()
        results = self.titleDict[u_title]
        # Check for matching year
        for i_index, i_year, i_genre, i_article, i_altTitle in results:
            if year == "":
                matches.append(i_index)
            elif year.replace(' ', '') in i_year:
                matches.append(i_index)
                print(matches)
        # Creative : Check if they used an alternative name
        # EDGE CASE: What if results aren't empty but they used a alt name?
        if len(results) == 0:
            try:
                movie_index = self.alternate_titles[u_title]
                matches.append(movie_index)
            except:
                pass
            # Creative Disambiguation (Part 1) Returns all movies containing the tokens in title
            if self.creative:
                for token_index, token in enumerate(title.split()):
                    preprocessed_token = token.lower()
                    try:
                        #Check the intersection of the keywords if there are multiple words
                        #if the code got something from the previous code it should add
                        if token_index == 0:
                            matches += self.disamb_title[preprocessed_token]
                        else:
                            token_movies = self.disamb_title[preprocessed_token]
                            common_movies = set(matches).intersection(set(token_movies))
                            matches = list(common_movies)
                    except:
                        pass
                
        # Removes duplicates
        matches = list(set(matches))
        return matches

    def extract_sentiment(self, preprocessed_input, returnFlip = False, startSentence = True):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        
        wordspattern = "((?:[A-Za-z])[\w']+)"
        

        # replace titles with X so it doesnt interfere with sentiment   
        preprocessed_input = re.sub(self.titlepattern, 'X', preprocessed_input)
        words = re.findall(wordspattern, preprocessed_input)

        # this shouldn't happen, if it does, there are bugs in non sentiment code
        if len(words) == 0:
            print("tell the idiot that coded this that there is an empty list being passed into the sentiment function")
            return 0
        # we want to check if the previous word is a flip, so start with the first word outside the loop
        flip = -1 if words[0] in self.flipWords and startSentence else 1
        total = 0 if self.stem(words[0]) not in self.newsentiment else self.newsentiment[self.stem(words[0])] # start with first word

        # if we are returning the flip we need to assume there is a sentiment before the first word
        for i in range(1, len(words)):
            currWord = self.stem(words[i])
            prevWord = self.stem(words[i - 1])
            # flip the sentiment if a flip word comes before it
            flip = -1 if prevWord in self.flipWords else flip if prevWord in self.fillerWords else 1
            if currWord in self.newsentiment:
                total +=  flip * self.newsentiment[currWord]
        if returnFlip:
            return total/max(1,abs(total)), flip
        return total/max(1,abs(total))

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        def cleanup(x):
            return x.replace(" ","").replace(":","").replace(",","").replace(".","")

        conjugations = ['and','but','&','or']
        opinions = []
        sentiments = []
        for conj in conjugations:
            if conj not in preprocessed_input.split(): continue
            opinions = preprocessed_input.split(conj)
    
        for i in range(len(opinions)):
            opinion = opinions[i]
            title = self.extract_titles(opinion)[0]
            opinion = opinion.split()
            
            if cleanup(re.sub(self.titlepattern, '', opinions[i])) == "":
                sentiments.append((title, sentiments[i-1][1]))
                continue
            
            sentiment, flip = self.extract_sentiment(opinions[i], True)
            if sentiment == 0: sentiments.append((title, flip * sentiments[i-1][1]))
            else: sentiments.append((title, sentiment))
            
            
            #sentiments.append((title, sentiment))
        #print(sentiments)
        return sentiments

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        # Calculates the min edit distance betweens two strings and 
        def calc_min_edit_dist(s1, s2):
            memo_array = np.zeros((len(movie_title) + 1,len(preproccessed_title) + 1))
            # Initialization
            memo_array[:,0] = np.array(range(len(s1) + 1))          
            memo_array[0,:] = np.array(range(len(s2) + 1))     
            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    add = memo_array[i - 1, j] + 1
                    delete = memo_array[i, j - 1] + 1
                    cost = 2
                    if s1[i - 1] == s2[j - 1]:
                        cost = 0
                    sub = memo_array[i - 1, j - 1] + cost
                    memo_array[i, j] = min([add, delete, sub])
            return memo_array[len(s1), len(s2)]
        
        potential_titles = []
        preproccessed_title = title.lower()
        min_distance = 10000000000000000000 # arbitrarily large value
        movie_options = self.titleDict.keys()
        for i in range(len(self.titles)):
            movie_title = self.titles[i][0].lower().split(' (')[0] 
            split_title = movie_title.split(', ')
            if len(split_title) > 1:
                movie_title = split_title[1] + ' ' + split_title[0]
            dist = calc_min_edit_dist(movie_title, preproccessed_title)
            if dist <= max_distance:
                if dist < min_distance:
                    potential_titles = [i]
                    min_distance = dist
                elif dist == min_distance:
                    potential_titles.append(i)
        return potential_titles
    
    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        # Iterate through the options and see which movie(s) are best
        options = []
        
        # Check if the clarification is just a number and represent the order in the list of candidates
        if clarification.isnumeric() and len(candidates) >= int(clarification) and int(clarification) != 0:
            return [candidates[int(clarification) - 1]]
        candlen = len(candidates)
        if clarification == 'most recent' and candlen >= 1: return [candidates[0]]

        places = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eight", "ninth","tenth"]
        for i, place in enumerate(places):
            if place in clarification and candlen >= i + 1: 
                print(place)
                return [candidates[i]]
        
        # if "second" in clarification: return [candidates[1]]
        # if "third" in clarification: return [candidates[2]]
        # if "fourth" in clarification: return [candidates[3]]
        # if "fifth" in clarification: return [candidates[4]]
        # if "sixth" in clarification: return [candidates[5]]
        # if "seventh" in clarification: return [candidates[6]]
        # if "eigth" in clarification: return [candidates[7]]
        # if "ninth" in clarification: return [candidates[8]]
        
        for movie_idx in candidates:
            title, genre = self.titles[movie_idx]
            pattern = "((?:[\w\.'é:\+\-\&/!?ó*\[\]]+\s?)+)(?:\s|,\s(.+)\s)?(?:\((.+)\)\s)?(?:\((\d\d\d\d)-?(?:\d\d\d\d)?\))"
            res = re.findall(pattern, title)
            if len(res) > 0:
                title, article, altTitle, year = re.findall(pattern, title)[0]
                #print(movie_idx, clarification, title, article, altTitle, year)
                # Check if the clarification is in the artle ex
                if clarification in title or clarification in article or clarification in altTitle or clarification == year:
                    options.append(movie_idx)
                    
        # If nothing is find lets get complex with it
        # We will split the strings into tokens as iterate through each string and see
        # Which movie has the highest frequency of common words
        if len(options) == 0:
            highest_common_words = 0
            best_canadiate = None
            for movie_option_index in candidates:
                common_words = 0
                for token in clarification.split():
                    movie_title = self.titles[movie_option_index][0]
                    if token in movie_title:
                        common_words += 1
                    if common_words > highest_common_words:
                        highest_common_words = common_words
                        best_canadiate = movie_option_index
            if best_canadiate != None: options.append(best_canadiate) 
        print("clarification: {}".format(clarification))
        print("candidates: {}".format(candidates))
        print("new candidates: {}".format(options))
        return options if len(options) > 0 else candidates

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.
        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binstep = np.where(ratings == 0, 100, ratings)
        binstep = np.where(binstep <= threshold, -1, binstep)
        binstep = np.where(binstep == 100, 0, binstep)
        binstep = np.where(binstep > threshold, 1, binstep)
        binarized_ratings = binstep

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        denom = (np.linalg.norm(u)* np.linalg.norm(v))
        if denom == 0:
            return 0
        similarity = np.dot(u, v)/denom
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the ratin g for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################
        # Populate this list with k movie indices to recommend to the user.
        userRatedIndices = np.nonzero(user_ratings)[0]
        movieRatings = np.zeros(ratings_matrix.shape[0])
        for i in range(len(movieRatings)):
            if i not in userRatedIndices: 
                for userRatedIndex in userRatedIndices:
                    similarity_score = self.similarity(ratings_matrix[userRatedIndex], ratings_matrix[i])
                    movieRatings[i] +=  similarity_score * user_ratings[userRatedIndex]
        top = np.flip(np.argsort(movieRatings)) # sort from highest to lowest
        recommendations = top[:k]
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return list(recommendations)

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')

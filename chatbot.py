# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re
import numpy as np
import collections
from porter_stemmer import PorterStemmer



# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'chris'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        # put self.titles in a better format
        def refineTitles(titleList):
            #print("preprocessing title data...")
            titles = collections.defaultdict(list)
            pattern = "((?:[\w\.'é:\+\-\&/!?ó*\[\]]+\s?)+)(?:\s|,\s(.+)\s)?(?:\((.+)\)\s)?(?:\((\d\d\d\d)-?(?:\d\d\d\d)?\))"
            for i in range(len(titleList)):
                titlefromlist, genre = titleList[i]
                #print("{}: {}".format(i, titlefromlist))
                res = re.findall(pattern, titlefromlist)
                if len(res) > 0:
                    title, article, altTitle, year = re.findall(pattern, titlefromlist)[0]
                    title = title.replace(" ", "").lower()
                    titles[title].append([i, year, genre, article, altTitle])
                else:
                    titles[titlefromlist].append([i, None, genre, None, None])
            #print("done processing data...")
            return titles
        def refineSentiment(sentiment):
            newSentiment = collections.defaultdict(int)
            def stem(x):
                stemmer = PorterStemmer()
                return stemmer.stem(x, 0, len(x) - 1)
            for key, value in sentiment.items():
                newSentiment[stem(key)] = 1 if value == "pos" else -1
            return newSentiment
        self.titleDict = refineTitles(self.titles)
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.newsentiment = refineSentiment(self.sentiment)

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

        greeting_message = "How can I help you?"

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

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

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
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

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
        # space_separated = text.split(' ')
        # arr = []
        # token = ""
        # for word in space_separated:
        #     if token != "":
        #         token += " " + word
        #         if word[-1] == '\"':
        #             arr.append(token)
        #             token = ""
        #     elif word[0] == '\"':
        #         token += word
        #     else:
        #         arr.append(word)
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
        pattern = "\"([\w\.'é:\-\&/!?ó*\[\]\(\) ]+)\""
        
        
        return re.findall(pattern, preprocessed_input)

        

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
        pattern = "(?:(?:([Aa]|[Tt]he|[Aa]n)) )?((?:(?:[\w.'é:\-\&/!?ó]+) ?)+)(?:\((\d\d\d\d)\))?"
        #print("xyz title: {}".format(title))
        article, u_title, year = re.findall(pattern, title)[0]
        u_title = u_title.replace(" ", "").lower()
        #print("xyz u_title: {}".format(u_title))
        results = self.titleDict[u_title]
        
        #print("xyz res: {}".format(results))
        for i_index, i_year, i_genre, i_article, i_altTitle in results:
            #print(year)
            #print(year == "")
            if year == "":
                matches.append(i_index)
            elif year == i_year:
                matches.append(i_index)


        return matches

    def extract_sentiment(self, preprocessed_input):
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
        def stem(x, islist = False):
            stemmer = PorterStemmer()
            if islist:
                return [stem(y) for y in x ]
            return stemmer.stem(x, 0, len(x) - 1)
        flipWords = stem(["not","didn't", "never"], True)
        fillerWords = stem(["really","absolutely", "undoubtedbly", "honestly"], True)
        titlepattern = "\"([\w\.'é:\-\&/!?ó*\[\]\(\) ]+)\""
        wordspattern = "((?:[A-Za-z])[\w']+)"
        flip = 1

        # replace titles with X so it doesnt interfere with sentiment   
        preprocessed_input = re.sub(titlepattern, 'X', preprocessed_input)
        words = re.findall(wordspattern, preprocessed_input)
        # we want to check if the last word is a flip, so start with the first word outside the loop
        total = 0 if stem(words[0]) not in self.newsentiment else self.newsentiment[stem(words[0])] # start with first word

        for i in range(1, len(words)):
            currWord = stem(words[i])
            prevWord = stem(words[i - 1])
            # flip the sentiment if a flip word comes before it
            flip = -1 if prevWord in flipWords else flip if prevWord in fillerWords else 1
            if currWord in self.newsentiment:
                total +=  flip * self.newsentiment[currWord]

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
        pass




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

        pass

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
        pass

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
        u_rated = np.nonzero(u)[0]
        v_rated = np.nonzero(v)[0]
        same = np.intersect1d(u_rated, v_rated)
        u = u[same]
        v = v[same]
        if len(u) < 1 or len(v) < 1:
            return 0
        similarity = np.dot(u, v)/(np.linalg.norm(u)* np.linalg.norm(v))
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
          `ratings_matrix[i, j]` is the rating for movie i by user j
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
            if i in userRatedIndices: continue
            
            for userRatedIndex in userRatedIndices:
                movieRatings[i] += self.similarity(ratings_matrix[userRatedIndex], ratings_matrix[i]) * user_ratings[userRatedIndex]
        
        top = np.argsort(movieRatings)
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

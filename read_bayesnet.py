"""
A .bif file reader using regular expressions
Currently only supports discrete domains
Freely usable and extensible
Author: Bob Ruiken
https://github.com/bornobob/BayesianNetwork-.bif-reader/blob/master/read_bayesnet.py
"""

import re
from pathlib import Path

# The regular expressions to parse the .bif file
NETWORK_RE = r' *network +((?:[^{]+ +)*[^{ ]+) *{\n+ *}'
VAR_RE = r' *variable +([^{ ]+) +{\n+ *type +(?:discrete|) +\[ +(\d+) +\] *{ *([a-zA-Z, 0-9]+)};\n+ *}'
PROB_RE = r' *probability +\( *(?P<var>[^ ]+) *(?:\| *(?P<parents>[^\)]+))?\) *{\n([^}]+)\n *}'
TABLE_RE = r' *table +([^;]+) *;'
PARENTS_RE = r' *\(([^\)]+)\) +([^;]+) *;'

FLAGS = re.RegexFlag.M  # Multi line flag

# Compiling the regular expressions with the flags
network_re = re.compile(NETWORK_RE, flags=FLAGS)
var_re = re.compile(VAR_RE, flags=FLAGS)
prob_re = re.compile(PROB_RE, flags=FLAGS)
table_re = re.compile(TABLE_RE, flags=FLAGS)
parents_re = re.compile(PARENTS_RE, flags=FLAGS)


class Variable:
    """
    A simple class to save variables into
    If self.parents is empty then the self.probabilities will be filled with the probabilities keyed on the domain
    Otherwise the self.probabilities will be filled with dictionaries keyed on the domain in dictionaries keyed on the
    assignments of the parents
    """
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.probabilities = {}
        self.parents = []
        self.markov_blanket = []

    def get_probability(self, assignment):
        """
        Returns the probability tuple for assignment given.
        For example, if the variable has one parent with domain ['True', 'False'], the parameter assignment should
        either look like ['True', 'True'].
        :param assignment: the assignment you want to get the probabilities for
        :return: the probabilities for the assignment as a dictionary keyed on the domain
        or KeyError if the assignment is not found
        """
        key = tuple(assignment)
        if key in self.probabilities.keys():
            return self.probabilities[key]
        else:
            raise KeyError('Key ({}) is not an assignment for this variable.'.format(key))


class BayesianNetwork:
    """
    The BayesianNetwork class stores the variables and can read a network from a .bif file
    """
    def __init__(self, file):
        self.file = file
        self.name = ''
        self.variables = []
        self.parse_file()

    def parse_file(self):
        """
        Parses the supplied file in the init function
        """
        # Read in the entire file
        contents = Path(self.file).read_text()

        # Find the name of the network
        self.parse_network(contents)

        # Find all the variables
        self.parse_variables(contents)

        # Find all the probabilities
        self.parse_probabilities(contents)
        
        # Define Markov Blanket
        self.define_markov_blankets()
        
        

    def parse_network(self, content):
        """
        Parses the network portion of the .bif file
        :param content: the content of the .bif file
        """
        network = network_re.match(content)
        try:
            self.name = network.group(1)
        except IndexError:
            self.name = 'Unnamed network'

    def parse_variables(self, content):
        """
        Parses the variables of the .bif file
        :param content: the content of the .bif file
        """
        variables = var_re.findall(content)
        for _name, _, _values in variables:
            domain = [x.strip() for x in _values.split(',')]
            self.variables.append(Variable(_name, domain))

    def parse_probabilities(self, content):
        """
        Parses the probabilities of the .bif file
        :param content: the content of the .bif file
        """
        probabilities = prob_re.findall(content)
        for _name, _parents, _probabilities in probabilities:
            if _parents:
                values = parents_re.findall(_probabilities)
                parents = [x.strip() for x in _parents.split(',')]
                var = self.get_variable(_name)
                var.parents = parents
                for val in values:
                    key = tuple([v.strip() for v in val[0].split(',')])
                    prob = [float(v.strip()) for v in val[1].split(',')]
                    var.probabilities[key] = dict(zip(var.domain, prob))
            else:
                value = table_re.match(_probabilities)
                var = self.get_variable(_name)
                var.probabilities = dict(zip(var.domain, [float(v.strip()) for v in value.group(1).split(',')]))

    def get_variable(self, name):
        """
        Returns the instance of the Variable class with name: name
        :param name: the name of the sought variable
        :return: the Variable instance
        """
        for var in self.variables:
            if var.name == name:
                return var
            
    
    
    
    '''
    The methods below where written for the assignment
    '''


    def define_markov_blankets(self):
        '''
        In a Bayesian network, the Markov blanket of a node includes its parents,
         children and the other parents of all of its children. However here we only
         take into account the nodes directly connected to a certain node, including itself.
        '''
        
        mb = dict()
        for v in self.variables: # for each variable
            mb[v.name] = set()
            mb[v.name].add(v.name) # add itself to the list
            
        for v in self.variables: # for each variable
            for p in v.parents: # for each parent
                mb[v.name].add(p) # add the parent
                mb[p].add(v.name) # add itself to the parent mb

        for v in self.variables:            
            v.markov_blanket = list(mb[v.name])
            
            
    def generate_sample(self,evidence):
        '''
        Generates a random value for each variable that is not evidence
        
        :param evidence: existing evidence
        :return: a dict
        '''
        
        s = dict()
        for v in self.variables:
            if v.name in evidence:
                s[v.name] = evidence[v.name]  
            else: 
                s[v.name] = random.choice(v.domain)
            
        return s
    

    def get_random_number(self):
        '''
        Returns a random float uniformly between 0 and 1
        
        :return: a random float
        '''
        return random.uniform(0,1) 
       
    
    def generate_monitor(self):
        '''
        Generates a dict where each key is a variable name and its value
        to be monitered throughout the iterations
        
        :return: a dictionary
        '''
        tmp = dict()
        for v in self.variables:
            tmp[v.name] = random.choice(v.domain)
        return tmp
    
    
    def gibbs_sampling(self,iterations=100, warm_up=1000, evidence={}, query=False, debug=False):
        '''
        The sampling algorithm.
        
        :param iterations: number of iterations
        :param warm_up: minimum number of iterations
        :param evidence: pre-existing evidence
        :param query: if it's a query calling the function or not
        :param debug: print extra stuff or not
        '''
        
        iterations += warm_up 
        
        sample = self.generate_sample(evidence)
        if debug: print('Initial sample :', sample)
            
        monitor = self.generate_monitor()
        if debug: print('Monitor :', monitor)
        
        # list of lists where each list contains 
        # probabilities for each iteration
        percentages = [] 

        # save the number of time a value has appeared 
        # in the sample for each variable
        results = dict() 
        
        # list of names
        columns = list()
       
        # initialize results dict
        for var in self.variables:
            columns.append(var.name)
            results[var.name] = dict()
            for d in var.domain:
                results[var.name][d] = 0
                
        i = 0 # current iteration number
        samples = dict() # all samples, keys are iteration number
        samples[i] = sample.copy() # add first sample
        
        while i < iterations: 
            if debug : print(i, end='\r')
            
            for v in sample: 
                
                if v not in evidence: 
                    
                    cur = self.get_variable(v)
                    mb = cur.markov_blanket

                    # list of probablities needed to calculate
                    # new value for current node
                    probs = [] 
                    
                    for node in mb: # for every node in the markov blanket
                        n = self.get_variable(node)
                        
                        if len(n.parents) == 0: # has no parents
                            
                            if v == n.name: #if its main node, get probabilities diretcly
                                probs.append(n.probabilities)  
                            else: # get probabilities using the sample
                                probs.append(n.probabilities[sample[n.name]])
                            
                        elif len(n.parents) > 0: # has parents
                            
                            
                            if v in n.parents: # main node is part of parents of markov blanket node
                                
                                # due to the structure of the probabilities object 
                                # we need to get the probabilities for all 
                                # possible values and then normalize
                                
                                nom = dict() # probabilites for each possible value
                                denom = 0.0
                                for d in cur.domain:
                                    tmp = []
                                    for par in n.parents:
                                        if par == v: # if parent is main node, get for possible domain d
                                            tmp.append(d)  
                                        else: # otherwise, get value from sample
                                            tmp.append(sample[par])
                                    
                                    # get probabilities for parents values
                                    nom[d] = n.get_probability(tmp)[sample[node]]
                                    
                                    # sum probabilities to the denominator
                                    denom += nom[d]
                                
                                # sometimes the doniminator is 0
                                # so in order to avoid division by 0
                                # turn that into 1
                                if denom == 0.0: 
                                    denom = 1.0  
                                    
                                #normalize values
                                for d in cur.domain: 
                                    nom[d] /= denom
                                
                                # add values to probabilities list
                                probs.append(nom)
                                        
                            else: # if main node is not part of parents
                                tmp = []
                                
                                # get parent values from sample
                                for par in n.parents:
                                    tmp.append(sample[par])
                                
                                # get probabilities according to parent values
                                if node == v:
                                    probs.append(n.get_probability(tmp))
                                else:
                                    probs.append(n.get_probability(tmp)[sample[node]])

                    denom = 0.0
                    
                    # dictionary containing probability of node for each value
                    d_dict = dict() 
                    
                    for d in cur.domain: # for each possible value of current node
                        tmp = 1.0 
                        for p in probs: # and for each probability 
                            
                            if type(p)==type({}): # if p is a dictonary, get probability for value d
                                tmp *= p[d]
                            else: # if p is a number, multiply directly
                                tmp *= p
                        
                        # probabilty for value d 
                        d_dict[d] = tmp
                        
                        # add that probabilty to the denominator
                        denom += tmp

                    # sometimes the doniminator is 0
                    # so in order to avoid division by 0
                    # turn that into 1    
                    if denom == 0.0: 
                        denom = 1.0
                    
                    # normalize
                    for d in d_dict:
                        d_dict[d] /= denom

                    # order probabilites from higher to lower    
                    d_dict = {k: v for k, v in sorted(d_dict.items(), key=lambda item: item[1], reverse=True)}

                    # sum values from higher to lower so last value is equal to 1
                    s_dict = d_dict.copy()
                    l_value = 0.0
                    for k in s_dict:
                        s_dict[k] += l_value
                        l_value = s_dict[k]

                    # update sample
                    rand = self.get_random_number()
                    for k in s_dict:
                        
                        # if random value fits in the probability, 
                        # update sample and leave loop
                        if rand <= s_dict[k]:
                            sample[v] = k
                            break
            
            i+=1
            
            # add new sample to sample list
            samples[i] = sample.copy()
            
            # update counts in results
            for var in sample:
                results[var][sample[var]] += 1
            
            # calculate probabilities for this iteration
            if not query:
                tmp = []
                for var in results:
                    tmp.append(results[var][monitor[var]] / i)
                percentages.append(tmp)
           

        df = pd.DataFrame(percentages, columns=columns)
        
        # calculate final probabilities
        for k in results:
            for v in results[k]:
                results[k][v] = round(results[k][v]/i,2)
                
        return results, df
            
# our imports            
import random
import pandas as pd
random.seed(7)
            
# bn = BayesianNetwork(file='earthquake.bif')  # example usage for the supplied earthquake.bif file
# for v in bn.variables:
#     print(v.probabilities)
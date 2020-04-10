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
    The methods listed below where written for the assignment
    '''


    def define_markov_blankets(self):
        '''
        In a Bayesian network, the Markov blanket of a node includes its parents,
         children and the other parents of all of its children.
        '''
        
        mb = dict()
        for v in self.variables:
            mb[v.name] = set()
            
            mb[v.name].add(v.name)


        for v in self.variables:

            for p in v.parents:
                mb[v.name].add(p)
                mb[p].add(v.name)
                #for p1 in v.parents:
                #    if p != p1:
                #        mb[p].add(p1)
                #        mb[p1].add(p)

        for v in self.variables:
            v.markov_blanket = list(mb[v.name])
            
            
    def generate_sample(self,evidence):
        s = dict()
        for v in self.variables:
            if v.name in evidence:
                s[v.name] = evidence[v.name]
            s[v.name] = random.choice(v.domain)
        return s
    
    
    def generate_numbers(self,size):
        size = size * len(self.variables)
        return [random.uniform(0,1) for x in range(size)]
        #return [random.random() for x in range(t_size)]
       
    
    def generate_monitor(self):
        tmp = dict()
        for v in self.variables:
            tmp[v.name] = random.choice(v.domain)
        return tmp
    
    
    def gibbs_sampling(self,iterations=10000, warm_up=100, evidence={}, query=False, debug=False):
        
        self.parse_file() # reset probabilities
        
        iterations += warm_up
        numbers = self.generate_numbers(iterations)
        sample = self.generate_sample(evidence)
        prev_sample = sample.copy()
        monitor = self.generate_monitor()
        #print(numbers)
        df = pd.DataFrame()
          
        print(sample)
          
        
        i = 0
        results = dict()
        
        samples = dict()
        samples[i] = sample.copy()
        
        while i < iterations: 
            print(i, end='\r')
            
            for v in sample: # for each random value in the sample
                if v not in evidence:
                    cur = self.get_variable(v)
                    mb = cur.markov_blanket

                    probs = []
                    for node in mb: # for every node in the markov blanket
                        n = self.get_variable(node)

                        if len(n.parents) == 0: # has no parents
                            probs.append(n.probabilities) if v == n.name else probs.append(n.probabilities[sample[n.name]])

                        elif len(n.parents) > 0: # has parents
                            
                            
                            #  {('yes',): {'yes': 0.98, 'no': 0.02}, ('no',): {'yes': 0.05, 'no': 0.95}} 
                            
                            
                            if v in n.parents: # main node is part of parents of markov blanket node
                                tmp1 = dict()
                                denom = 0.0
                                for d in cur.domain:
                                    tmp = []
                                    for par in n.parents:
                                        tmp.append(d) if par == v else tmp.append(sample[par])
                                        
                                    tmp1[d] = n.get_probability(tmp)[sample[node]]
                                    denom += tmp1[d]
                                
                                #if denom == 0.0 : denom = 1.0  
                                    
                                for d in cur.domain: 
                                    tmp1[d] /= denom
                                probs.append(tmp1)
                                        
                            else:
                                tmp = []
                                for par in n.parents:
                                    tmp.append(sample[par])
                                
                                probs.append(n.get_probability(tmp))
                                
                    #print(v)
                    #print(probs)

                    denom = 0.0
                    d_dict = dict()
                    for d in cur.domain:
                        tmp = 1.0
                        for p in probs:
                            if type(p)==type({}): # if p is a dictonary, get a value
                                tmp *= p[d]
                            else:
                                tmp *= p
                        d_dict[d] = tmp
                        denom += tmp


                    #print(v)
                    #print('antes de normalizar',d_dict)
                    #if denom == 0.0: denom = 1.0
                    for d in d_dict:
                        d_dict[d] /= denom
                    #print('depois de normalizar',d_dict)    

                    d_dict = {k: v for k, v in sorted(d_dict.items(), key=lambda item: item[1], reverse=True)}
                    #print('depois de ordenar',d_dict)

                    s_dict = d_dict.copy()
                    l_value = 0.0
                    for k in s_dict:
                        s_dict[k] += l_value
                        l_value = s_dict[k]
                    #print('depois de somar',s_dict,'\n\n')

                    # update sample
                    rand = numbers.pop()
                    for k in s_dict:
                        if rand <= s_dict[k]:
                            sample[v] = k
                    
            print(sample)
            i+=1
            
            samples[i] = sample.copy()     
            
            results = dict()
            for var in self.variables:
                new_prob = dict()
                for d in var.domain:
                    new_prob[d] = 0.0
                    for samp in samples:
                        if samples[samp][var.name] == d:
                            new_prob[d] += 1.0
                    new_prob[d] /= len(samples)
                results[var.name] = new_prob.copy()
              
            # for plots
            if not query:
                df = df.append(results.copy(),ignore_index=True)
                
        if not query:
            for m in monitor:
                df[m] = df[m].apply(lambda x : x[monitor[m]])
                #df = df.applymap(lambda x : x['yes'])
        
        for k in results:
            for v in results[k]:
                results[k][v] = round(results[k][v],2)
                
        return results, df
    
    
    def exists(self,name):
        exists = False
        for v in self.variables:
            if name == v.name:
                return True
        return False
    
    
    def has_domain(self,name,value):
        v = self.get_variable(name)
        if value in v.domain:
            return True
        return False
    
    
    def similar(self,a, b):
        return SequenceMatcher(None, a, b).ratio()
            
            
                
from scipy.spatial import distance
import random
import pandas as pd
from difflib import SequenceMatcher
random.seed(7)
            
# bn = BayesianNetwork(file='earthquake.bif')  # example usage for the supplied earthquake.bif file
# for v in bn.variables:
#     print(v.probabilities)
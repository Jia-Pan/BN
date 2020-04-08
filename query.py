from read_bayesnet import BayesianNetwork, Variable
from os import path

def main():
    
    print('Welcome!')
    while(True):
        print('Please insert the path to the bif file')
        file = input('Press [ENTER] for asia.bif or path -> ')
        if file == '':
            file = 'asia.bif'
        if not path.exists(file):
            print('File not found!')
        else: 
            print('File Found! Loading...',end='')
            break
        
    try:
        bn = BayesianNetwork(file='asia.bif')
        print('Done\nTry a query! ex: Pr? asia  or  Pr? asia|dysp=yes')
    except e:
        print('\nSomething went wrong. Please try another file')
        print(e)
        
    
    while(True):
        query = input()
        query = query.replace('Pr? ','')
        q = query.split('|')
        
        if len(q) == 1: #no evidence
            if not bn.exists(q[0]):
                print('Variable not found')
            else:
                result, _ = bn.gibbs_sampling(query=True)
                result = result[q[0]]
                print(result)
                
        elif len(q) > 1: # has evidence
            evidence = q[1].split(',')
            e_dict = dict()
            for e in evidence:
                
                e1 = e.split('=')
                if bn.exists(e1[0]):
                    if bn.has_domain(e1[0],e1[1]):
                        e_dict[e1[0]] = e1[1]
                    else:
                        print('Not a valid value for variable {}'.format(e[0]))
                else:
                    print('Variable not found')
            
            result, _ = bn.gibbs_sampling(evidence=e_dict,query=True)
            result = result[q[0]]
            print(result)
                
        else:
            print('Not a valid query')
    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nBye')
import json
import random
import numpy as np

class DBQuery():
    
    def __init__(self, data_dir, cfg):
        # loading databases
        self.cfg = cfg
        self.dbs = {}
        for domain in cfg.belief_domains:
            with open('{}/{}_db.json'.format(data_dir, domain)) as f:
                self.dbs[domain] = json.load(f)
    
    def query(self, domain, constraints, ignore_open=True):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state"""
        # query the db
        if domain == 'taxi':
            return [{'taxi_type': random.choice(self.dbs[domain]['taxi_colors']) + ' ' + random.choice(self.dbs[domain]['taxi_types']),
                     'taxi_phone': ''.join([str(random.randint(0, 9)) for _ in range(10)])}]
        elif domain == 'hospital':
            return self.dbs['hospital']
        elif domain == 'police':
            return self.dbs['police']
    
        found = []
        for i, record in enumerate(self.dbs[domain]):
            for key, val in constraints:
                if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    try:
                        record_keys = [key.lower() for key in record]
                        if key.lower() not in record_keys:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        else:
                            if val.strip() != record[key].strip():
                                break
                    except:
                        continue
            else:
                record['ref'] = f'{domain}-{i:08d}'
                found.append(record)
    
        return found
    
    def pointer(self, turn, mapping, db_domains, requestable, noisy):
        """Create database pointer for all related domains."""        
        pointer_vector = np.zeros(6 * len(db_domains))
        entropy = np.zeros(len(requestable))
        for domain in db_domains:
            constraint = []
            for k, v in turn[domain].items():
                if k in mapping[domain] and v != '?':
                    constraint.append((mapping[domain][k], v))
            entities = self.query(domain, constraint, noisy)
            pointer_vector = self.one_hot_vector(len(entities), domain, pointer_vector, db_domains)
            entropy = self.calc_slot_entropy(entities, domain, entropy, requestable)
    
        return pointer_vector, entropy
    
    def one_hot_vector(self, num, domain, vector, db_domains):
        """Return number of available entities for particular domain."""
        if domain != 'train':
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        else:
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num <= 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num <= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num <= 10:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num <= 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num > 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    
        return vector  
    
    def calc_slot_entropy(self, entities, domain, vector, requestable):
        """Calculate entropy of requestable slot values in results"""
        N = len(entities)
        if not N:
            return vector
        
        # Count the values
        value_probabilities = {}   
        for index, entity in enumerate(entities):
            if index == 0:
                for key, value in entity.items():
                    if key in self.cfg.map_inverse[domain] and \
                        domain+'-'+self.cfg.map_inverse[domain][key] in requestable:
                        value_probabilities[key] = {value:1}
            else:
                for key, value in entity.items():
                    if key in value_probabilities:
                        if value not in value_probabilities[key]:
                            value_probabilities[key][value] = 1
                        else:
                            value_probabilities[key][value] += 1
                        
        # Calculate entropies
        for key in value_probabilities:
            entropy = 0
            for count in value_probabilities[key].values():
                entropy -= count/N * np.log(count/N)
            vector[requestable.index(domain+'-'+self.cfg.map_inverse[domain][key])] = entropy
            
        return vector
    
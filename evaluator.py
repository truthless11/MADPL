import re
import numpy as np
from copy import deepcopy

from dbquery import DBQuery

informable = \
{'attraction': ['area', 'name', 'type'],
 'restaurant': ['addr', 'day', 'food', 'name', 'people', 'price', 'time'],
 'train': ['day', 'people', 'arrive', 'leave', 'depart', 'dest'],
 'hotel': ['area', 'day', 'internet', 'name', 'parking', 'people', 'price', 'stars', 'stay', 'type'],
 'taxi': ['arrive', 'leave', 'depart', 'dest'],
 'hospital': ['department'],
 'police': []}

requestable = \
{'attraction': ['post', 'phone', 'addr', 'fee', 'area', 'type'],
 'restaurant': ['addr', 'phone', 'post', 'price', 'area', 'food'],
 'train': ['ticket', 'time', 'id', 'arrive', 'leave'],
 'hotel': ['addr', 'post', 'phone', 'price', 'internet', 'parking', 'area', 'type', 'stars'],
 'taxi': ['car', 'phone'],
 'hospital': ['phone'],
 'police': ['addr', 'post']}

time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
NUL_VALUE = ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care"]

class MultiWozEvaluator():
    def __init__(self):
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = {}
        self.booked = {}
        self.cur_domain = ''
        self.complete_domain = []
        from config import MultiWozConfig
        cfg = MultiWozConfig()
        self.belief_domains = cfg.belief_domains
        self.mapping = cfg.mapping
        db = DBQuery('multiwoz', cfg)
        self.dbs = db.dbs

    def _init_dict(self):
        dic = {}
        for domain in self.belief_domains:
            dic[domain] = {}
        return dic

    def _init_dict_booked(self):
        dic = {}
        for domain in self.belief_domains:
            dic[domain] = None
        return dic

    def add_goal(self, goal):
        """
        init goal and array
        args:
            goal: dict[domain] dict[slot] value
        """
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = deepcopy(goal)
        for domain in self.belief_domains:
            if 'final' in self.goal[domain]:
                for key in self.goal[domain]['final']:
                    self.goal[domain][key] = self.goal[domain]['final'][key]
                del(self.goal[domain]['final'])
        self.cur_domain = ''
        self.complete_domain = []
        self.booked = self._init_dict_booked()

    def add_sys_da(self, da_turn):
        """
        add sys_da into array
        args:
            da_turn: dict[domain-intent-slot-p] value
        """
        for da_w_p in da_turn:
            domain, intent, slot, p = da_w_p.split('-')
            value = str(da_turn[da_w_p])
            da = '-'.join([domain, intent, slot])
            self.sys_da_array.append(da+'-'+value)

            if value != 'none':
                if da == 'booking-book-ref':
                    book_domain, ref_num = value.split('-')
                    if not self.booked[book_domain] and re.match(r'^\d{8}$', ref_num):
                        self.booked[book_domain] = self.dbs[book_domain][int(ref_num)]
                elif da == 'train-offerbooked-ref' or da == 'train-inform-ref':
                    ref_num = value.split('-')[1]
                    if not self.booked['train'] and re.match(r'^\d{8}$', ref_num):
                        self.booked['train'] = self.dbs['train'][int(ref_num)]
                elif da == 'taxi-inform-car':
                    if not self.booked['taxi']:
                        self.booked['taxi'] = 'booked'

    def add_usr_da(self, da_turn):
        """
        add usr_da into array
        args:
            da_turn: dict[domain-intent-slot] value
        """
        for da in da_turn:
            domain, intent, slot = da.split('-')
            value = str(da_turn[da])
            self.usr_da_array.append(da+'-'+value)
            if domain in self.belief_domains and domain != self.cur_domain:
                self.cur_domain = domain

    def _match_rate_goal(self, goal, booked_entity, domains=None):
        """
        judge if the selected entity meets the constraint
        """
        if domains is None:
            domains = self.belief_domains
        score = []
        for domain in domains:
            if 'book' in goal[domain]:
                tot = 0
                for key, value in goal[domain].items():
                    if value != '?':
                        tot += 1
                entity = booked_entity[domain]
                if entity is None:
                    score.append(0)
                    continue
                if domain in ['taxi', 'hospital', 'police']:
                    score.append(1)
                    continue
                match = 0
                for k, v in goal[domain].items():
                    if v == '?':
                        continue
                    if k in ['dest', 'depart', 'name'] or k not in self.mapping[domain]:
                        tot -= 1
                    elif k == 'leave':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arrive':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[self.mapping[domain][k]].strip():
                            match += 1
                if tot != 0:
                    score.append(match / tot)
        return score

    def _inform_F1_goal(self, goal, sys_history, domains=None):
        """
        judge if all the requested information is answered
        """
        if domains is None:
            domains = self.belief_domains
        inform_slot = {}
        for domain in domains:
            inform_slot[domain] = set()
        TP, FP, FN = 0, 0, 0
        for da in sys_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and \
                domain in domains and value.strip() not in NUL_VALUE:
                inform_slot[domain].add(slot)
        for domain in domains:
            for k, v in goal[domain].items():
                if v == '?':
                    if k in inform_slot[domain]:
                        TP += 1
                    else:
                        FN += 1
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal[domain] \
                and (k in requestable[domain] or k == 'ref'):
                    FP += 1
        return TP, FP, FN
    
    def _inform_F1_goal_usr(self, goal, usr_history, domains=None):
        """
        judge if all the constraint/request information is expressed
        """
        if domains is None:
            domains = self.belief_domains
        inform_slot = {}
        request_slot = {}
        for domain in domains:
            inform_slot[domain] = set()
            request_slot[domain] = set()
        TP, FP, FN = 0, 0, 0
        for da in usr_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent == 'inform':
                inform_slot[domain].add(slot)
            elif intent == 'request':
                request_slot[domain].add(slot)
        for domain in domains:
            for k, v in goal[domain].items():
                if v == '?':
                    if k in request_slot[domain]:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if k in inform_slot[domain]:
                        TP += 1
                    else:
                        FN += 1
            for k in inform_slot[domain]:
                if k not in goal[domain] \
                and k in informable[domain]:
                    FP += 1
            for k in request_slot[domain]:
                if k not in goal[domain] \
                and (k in requestable[domain] or k == 'ref'):
                    FP += 1
        return TP, FP, FN

    def _check_value(self, key, value):
        if key == "area":
            return value.lower() in ["centre", "east", "south", "west", "north"]
        elif key == "arriveBy" or key == "leaveAt":
            return time_re.match(value)
        elif key == "day":
            return value.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday",
                              "saturday", "sunday"]
        elif key == "duration":
            return 'minute' in value
        elif key == "internet" or key == "parking":
            return value in ["yes", "no"]
        elif key == "phone":
            return re.match(r'^\d{11}$', value)
        elif key == "price" or key == "entrance fee":
            return 'pound' in value or value in ["free", "?"]
        elif key == "pricerange":
            return value in ["cheap", "expensive", "moderate", "free"]
        elif key == "postcode":
            return re.match(r'^cb\d{2,3}[a-z]{2}$', value)
        elif key == "stars":
            return re.match(r'^\d$', value)
        elif key == "trainID":
            return re.match(r'^tr\d{4}$', value.lower())
        else:
            return True

    def match_rate(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self.goal
        else:
            goal = self._init_dict()
            for domain in self.belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = True
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d in self.belief_domains and i == 'inform'\
                    and s in informable[d]:
                    goal[d][s] = v
        score = self._match_rate_goal(goal, self.booked)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def inform_F1(self, ref2goal=True, ansbysys=True, aggregate=True):
        if ref2goal:
            goal = self.goal
        else:
            goal = self._init_dict()
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d in self.belief_domains and s in informable[d]:
                    if i == 'inform':
                        goal[d][s] = v
                    elif i == 'request':
                        goal[d][s] = '?'
        if ansbysys:
            TP, FP, FN = self._inform_F1_goal(goal, self.sys_da_array)
        else:
            TP, FP, FN = self._inform_F1_goal_usr(goal, self.usr_da_array)
        if aggregate:
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
        else:
            return [TP, FP, FN]

    def task_success(self, ref2goal=True):
        """
        judge if all the domains are successfully completed
        """
        book_sess = self.match_rate(ref2goal)
        inform_sess = self.inform_F1(ref2goal)
        # book rate == 1 & inform recall == 1
        if (book_sess == 1 and inform_sess[1] == 1) \
        or (book_sess == 1 and inform_sess[1] is None) \
        or (book_sess is None and inform_sess[1] == 1):
            return 1
        else:
            return 0

    def domain_success(self, domain, ref2goal=True):
        """
        judge if the domain (subtask) is successfully completed
        """
        if domain not in self.goal:
            return None
        if domain in self.complete_domain:
            return 0

        if ref2goal:
            goal = {}
            goal[domain] = deepcopy(self.goal[domain])
        else:
            goal = self._init_dict()
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if s in self.mapping[d]:
                    if i == 'inform':
                        goal[d][s] = v
                    elif i == 'request':
                        goal[d][s] = '?'

        match_rate = self._match_rate_goal(goal, self.booked, [domain])
        match_rate = np.mean(match_rate) if match_rate else None

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        try:
            inform_rec = inform[0] / (inform[0] + inform[2])
        except ZeroDivisionError:
            inform_rec = None

        if (match_rate == 1 and inform_rec == 1) \
        or (match_rate == 1 and inform_rec is None) \
        or (match_rate is None and inform_rec == 1):
            self.complete_domain.append(domain)
            return 1
        else:
            return 0
